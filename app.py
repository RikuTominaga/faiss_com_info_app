from flask import Flask, redirect, render_template, request, session, url_for

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import faiss

# 1. SQLiteデータベースの準備
def create_faq_database():
    conn = sqlite3.connect("./faq.db") #データベースにアクセス（存在しない場合はデータベースを作成）
    cursor = conn.cursor() #データベースに命令するための変数
    cursor.execute("CREATE TABLE IF NOT EXISTS faq (id INTEGER PRIMARY KEY,corporate_number TEXT,name TEXT ,title TEXT)") #もしもfaqという表がなければ作成する．

# サンプルデータをsample_faqsに入れる 下記のデータをデータベース中のfaqの項目questionに入れていく
    # title = [
    #     "How do I reset my password?", #文字列1
    #     "What payment methods do you accept?", #文字列2
    #     "Can I return an item I bought?", #文字列3
    #     "How can I track my order?" #文字列4
    # ]

    # name=["A","S","C","G"]

    # corporate_number =["01", "02", "03", "04"]

    # cursor.executemany("INSERT INTO faq (corporate_number,name,title) VALUES (?,?,?)", [(c,n,q) for c,n,q in zip(corporate_number,name,title)]) #sample_faqsから一行ずつ変数qとして取り出し，qをquestionに入れていく.executemanyは複数の命令を一度に実行するもの．
    conn.commit() #実行が反映


    return conn #データベースのオブジェクト変数を戻り値とする．あとでこのデータベースからデータを取り出すため．

# 2. 文章の埋め込みの作成
def compute_faq_embeddings(faq_questions, model):
    return np.array(model.embed_documents(faq_questions)) #model.encodeで文字列をベクトル化する

# 3. Faissインデックスの作成　Faissとはベクトル専用のデータベース
def create_faiss_index(model, embeddings, doc_ids):
    dimension=768
    index_flat = faiss.IndexFlatIP(dimension) #指定次元のベクトルを格納するFaissを整備.ベクトルの距離（コサイン）を計算できるよう初期化
    index = faiss.IndexIDMap(index_flat) #IDをベクトルに付与する（マッピング）
    index.add_with_ids(embeddings, doc_ids) #実際に文書のベクトルにIDを振ってデータベースに格納
    return index, index_flat

# 4. クエリ処理
def search_faq(query, model, index, k=3):
    query_embedding = model.embed_query(query) #質問文をベクトル化
    distances, indices = index.search(np.array([query_embedding],dtype=np.float32), k) #Faissデータベースから類似している文のトップ3を取得．kはトップkまで取得する意味
    return distances[0],indices[0]

# 5. 結果の取得
def get_faq_results(faq_ids, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faq WHERE id IN ({})".format(",".join("?" * len(faq_ids))), faq_ids) #結果をデータベースから取得
    return cursor.fetchall()

# データベース作成
conn = create_faq_database() #connはデータベースのオブジェクト変数
com_list=[]

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        #create_faq_database()
        conn = sqlite3.connect("./faq.db") #データベースにアクセス（存在しない場合はデータベースを作成）
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM faq") #faqから文をすべて取得する
        faq_data = cursor.fetchall() #結果をリストに格納

        print(faq_data)

        global com_list
        com_list=[]

        faq_ids, corporate_number, name, faq_questions = zip(*faq_data) #リストをIDとテキストに分ける
        model=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

        embeddings=compute_faq_embeddings(faq_questions, model)
        index, index_flat = create_faiss_index(model, embeddings, faq_ids)

        # クエリを入力して検索
        #query="What is your payment method"
        query=(request.form['input_query'])
        faq_distances,faq_indices = search_faq(query, model, index_flat,k=3) #質問文と類似している文を獲得

        # 結果を取得して表示
        results = get_faq_results([faq_ids[i] for i in faq_indices], conn)
        print("Results for query:", query)
        for r,d in zip(results,faq_distances):
            print(f"ID: {r[0]}, Corporate_number: {r[1]},name: {r[2]} , title: {r[3]}, Simularity:{d}")
            com_list.append([r[1],r[2],r[3],d])

        #session['name'] = str(r[1])
        #session['Question'] = str(r[2])
        #session['simularity'] = str(d)

        # データベース接続を閉じる
        # conn.close()

        
        session['output_query']=query

        return redirect(url_for('output'))
    return  render_template('input.html')

@app.route('/output')
def output():
    return render_template('output.html',arr=com_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True, port = 8000)