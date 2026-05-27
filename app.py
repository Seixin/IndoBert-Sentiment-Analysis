import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import re
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
from nlp_id.postag import PosTag
import nltk

st.set_page_config(page_title="Gadget Sentiment Analyzer", layout='wide')

@st.cache_resource
def siapkan_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

siapkan_nltk()

@st.cache_resource
def load_model():
    model_path = 'MichaelJo23/indobert-sentimen-gadgetin'
    nlp =pipeline(
        'text-classification',
        model=model_path,
        tokenizer=model_path,
        device=-1
    )
    return nlp

try:
    nlp_pipeline = load_model()
except Exception as e:
    st.error(f'Error Loading Model: {e}')

API_KEY = st.secrets['YOUTUBE_API_KEY']
youtube =build('youtube', 'v3', developerKey=API_KEY)

def ambil_komentar_yt(video_id, max_results=100):
    semua_komentar =[]
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId = video_id,
            maxResults = 100,
            textFormat = 'plainText'
        )
        jumlah_diambil = 0
        while request is not None and jumlah_diambil<max_results:
            response = request.execute()
            for item in response['items']:
                komentar = item['snippet']['topLevelComment']['snippet']['textDisplay']
                semua_komentar.append(komentar)
                jumlah_diambil +=1
                if jumlah_diambil >= max_results:
                    break

                if 'nextPageToken' in response and jumlah_diambil <max_results:
                    request= youtube.commentThreads().list_next(
                        previous_request=request,
                        previous_response=response
                    )
                else:
                    break
    except Exception as e:
        st.error(f'Gagal Menarik data API: {e}')

    return semua_komentar

st.title("Gadget Sentiment Analyzer")
tab1, tab2 = st.tabs(['Analisis Video Youtube', 'Uji Coba Manual'])

with tab1:
    st.subheader("Masukkan Link Video YouTube")
    url_input = st.text_input("URL Video:")
    batas_komentar = st.slider("Jumlah komentar yang ditarik:", 20, 200, 100, 20)
    
    if st.button("Analisis Video!"):
        if url_input:
            with st.spinner("Menarik data dan membaca sentimen..."):
                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url_input)
                if video_id_match:
                    video_id = video_id_match.group(1)
                    komentar_list = ambil_komentar_yt(video_id, max_results=batas_komentar)
                    
                    if komentar_list:
                        hasil_prediksi = nlp_pipeline(komentar_list, truncation=True, max_length=512)
                        df_hasil = pd.DataFrame({
                            "Komentar": komentar_list,
                            "Label": [res['label'] for res in hasil_prediksi]
                        })
                        
                        st.success(f"Berhasil menganalisis {len(df_hasil)} komentar!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Distribusi Sentimen")
                            pie_data = df_hasil['Label'].value_counts()
                            warna = {'Netral': '#66b3ff', 'Positif': '#99ff99', 'Negatif': '#ff9999'}
                            colors = [warna.get(x, '#808080') for x in pie_data.index]
                            fig, ax = plt.subplots()
                            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=colors, startangle=90)
                            ax.axis('equal')  
                            st.pyplot(fig)
                            
                        with col2:
                            st.write("### Word Cloud")
                            semua_teks = " ".join(df_hasil['Komentar'])
                            
                            postagger = PosTag()
                            hasil_pos = postagger.get_pos_tag(semua_teks)
                            
                            kumpulan_kata_sifat = [kata.lower() for kata, tag in hasil_pos if tag == 'JJ']
                            teks_kata_sifat = " ".join(kumpulan_kata_sifat)
                            
                            kata_abaikan = set(STOPWORDS)
                            stopwords_indo = [
                                "bisa", "pasti", "bener", "benar", "baru", "sama", "sangat", "banget", "paling"
                            ]
                            kata_abaikan.update(stopwords_indo)
                            
                            if teks_kata_sifat.strip():
                                wordcloud = WordCloud(stopwords=kata_abaikan, width=800, height=400, background_color='white', colormap='magma').generate(teks_kata_sifat)
                                fig2, ax2 = plt.subplots()
                                ax2.imshow(wordcloud, interpolation='bilinear')
                                ax2.axis("off")
                                st.pyplot(fig2)
                            else:
                                st.warning("Tidak ada kata sifat spesifik yang terdeteksi.")
                            
                        st.dataframe(df_hasil, use_container_width=True)
                else:
                    st.error("Format URL tidak valid.")

with tab2:
    st.subheader("Uji Model dengan Komentar Manual")
    teks_manual = st.text_area("Ketik komentar:")
    
    if st.button("Tebak Sentimen"):
        if teks_manual.strip():
            hasil = nlp_pipeline(teks_manual)[0]
            label = hasil['label']
            skor = hasil['score'] * 100
            
            if label == "Positif": st.success(f"**{label}** (Keyakinan: {skor:.2f}%)")
            elif label == "Negatif": st.error(f"**{label}** (Keyakinan: {skor:.2f}%)")
            else: st.info(f"**{label}** (Keyakinan: {skor:.2f}%)")
