import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules
import tkinter as tk


# CSV dosyalarını okuma
scoresdf = pd.read_csv('archive/genome_scores.csv')
genome_tagsdf = pd.read_csv('archive/genome_tags.csv')
##     link.csv dosyasına bu projede ihtiyacımız yok
moviedf = pd.read_csv('archive/movie.csv')
ratingdf = pd.read_csv('archive/rating.csv')
tagUserdf = pd.read_csv('archive/tag.csv')


###############################################################            Veri Temizleme işlemi

#############    genome_scores.csv, genome_tags.csv ve movie.csv dosyalarında gereksiz sütun yok

#### rating.csv dosyasında 'timestamp' sütununu silme
## silmek istenen 'timestamp' sütununu silme
'''ratingdf = ratingdf.drop(columns=["timestamp"])
ratingdf.to_csv('archive/rating.csv', index=False)'''

## tag.csv dosyasında 'timestamp' sütununu silme
'''tagUserdf = tagUserdf.drop(columns=["timestamp"])
tagUserdf.to_csv('archive/tag.csv', index=False)'''
## movie içinde genres sütunu boş olanları silme
moviedf = moviedf[moviedf['genres'] != '(no genres listed)']  # (no genres listed) çıkarma
moviedf = moviedf[moviedf['genres'] != '']                    # Boş string çıkarma
moviedf = moviedf[moviedf['genres'].notna()]                 # NaN değerleri çıkarma


####### ratingdfde sadece 1 film izlemiş kullanıcıları ve 1 kez izlenmiş filmleri silme

def FilteringRatingDF(ratingdf):
    # Film başına izlenme sayısını hesaplama
    movieCounts = ratingdf['movieId'].value_counts()

    # Sadece bir kez izlenmiş film id'lerini seçme
    singleWatchMovies = movieCounts[movieCounts == 1].index

    # Bu film id'lerini ratingdf'den çıkararak yeni bir DataFrame oluşturma
    ratingdfFiltered = ratingdf[~ratingdf['movieId'].isin(singleWatchMovies)]
    
    # Kullanıcı başına izlenme sayısını hesaplama
    userWatchCounts = ratingdfFiltered['userId'].value_counts()

    # Sadece bir film izlemiş kullanıcıları seçme
    singleWatchUsers = userWatchCounts[userWatchCounts == 1].index

    # Bu kullanıcıları ratingdf'den çıkararak yeni bir DataFrame oluşturma
    ratingdf = ratingdfFiltered[~ratingdfFiltered['userId'].isin(singleWatchUsers)]

    return ratingdf

ratingdf = FilteringRatingDF(ratingdf)



########################################################   En çok sevilen türleri seçme ve csv oluşturma

def CountGenres(df):               # Film Türlerini sayılarına ayıran fonksiyon
    df = df[df['genres'] != '(no genres listed)'] ## film türü belirtilmeyen filmleri silme işlemi
    # sütun içindeki türleri ayırma
    genresList = df['genres'].str.split('|').explode()
    
    ##genresList içindeki maddelerin sayar
    genresCount = genresList.value_counts()
    ###genres.csv adında yeni csv oluşturma
    genresdf = genresCount.reset_index()
    genresdf.columns = ['genres', 'count']

    return genresdf

genresdf = CountGenres(moviedf)  ##  movidedf içindeki film türlerini ayırma
#genresdf.to_csv('archive/genres.csv', index=False)  ##csv dosyasını oluşturma





########################################### Türlerdeki en iyi 50 filmi seçme ve csv dosyası oluşturma

genresAll = genresdf.iloc[:20, 0].str.lower() # Tüm film türlerini alma
genres10 = genresdf.iloc[:10, 0].str.lower() # en çok izlenen ilk on film türünü alma


#film türlerinin en alakalı türü bulma fonksiyonu
def FindMainTag(moviedf, scoresdf, tagdf):
    # Küçük harfe çevir
    tagdf['tag'] = tagdf['tag'].str.lower()
    
    # İlgili tag'ları filtrele
    tagdf = tagdf[tagdf['tag'].isin(genresAll)]

    # scoresdf ile moviedf'yi birleştir
    mergeddf = pd.merge(scoresdf, moviedf, on='movieId')

    # Relevance 0.5'in üstünde olanları filtrele
    filtereddf = mergeddf[mergeddf['relevance'] > 0.5]

    # En alakalı tag'ı bul (en yüksek relevance değeri olan)
    mainTags = filtereddf.loc[filtereddf.groupby('movieId')['relevance'].idxmax()][['movieId', 'tagId']]
    
    # main_tags ile tag_df'yi birleştirerek tag'ları al
    mainTags = mainTags.merge(tagdf, on='tagId', how='left')[['movieId', 'tag']]


    
    # movie_df ile main_tags'i birleştirerek mainTag sütununu ekle
    moviedf = moviedf.merge(mainTags, on='movieId', how='left').rename(columns={'tag': 'mainTag'})

    # NaN değerleri kontrol etme ve doldurma
    moviedf['mainTag'] = moviedf['mainTag'].fillna('No Tag Found')  
    
    # "No Tag Found" olanları filtreleme
    resultdf = moviedf[moviedf['mainTag'] != 'No Tag Found']

    # Eğer sonuç DataFrame'i boş değilse, döndür
    if not resultdf.empty:
        return resultdf.reset_index(drop=True)
    else:
        return None

moviedf = FindMainTag(moviedf, scoresdf, genome_tagsdf)
#moviedf.to_csv('archive/movieUpdated.csv', index=False)

# filmlerin idlerini eşleştirip rating ve movie dflerini birleştirme
movieUserdf = pd.merge(ratingdf, moviedf, on='movieId')

#movieUserdf.to_csv('archive/movieUser.csv', index=False)

def AddNewColForMovie():    
    df= movieUserdf.groupby('movieId').agg(
        #filmin ve türün ismine bir defa ihtiyacımız olduğu için first işlemini kullanma
        title = ('title', 'first'), 
        genres = ('genres', 'first'),
        mainTag = ('mainTag', 'first'),
        numOfRatings = ('rating', 'count'), #izlenme sayısı
        avgOfRating = ('rating', 'mean') #Ortalama derecesi
    ).reset_index() # tekrar indexli hale getirmek için

    return df


moviedf = AddNewColForMovie()
#moviedf.to_csv('archive/movieUpdated.csv', index=False)





def top50moviesByGenre(genre): #İzlenme sayısı ve ortalama ratingi hesaplama
    genreOfMoviedf = movieUserdf[movieUserdf['mainTag'].str.lower() == genre.lower()] #İstenen türle eşit olan filmleri seçtik
    
    genreOfMovieStats = genreOfMoviedf.groupby('movieId').agg(
        #filmin ve türün ismine bir defa ihtiyacımız olduğu için first işlemini kullandık
        title = ('title', 'first'), 
        genres = ('genres', 'first'),
        mainTag = ('mainTag', 'first'),
        numOfRatings = ('rating', 'count'), #izlenme sayısı
        avgOfRating = ('rating', 'mean') #Ortalama derecesi
    ).reset_index() # tekrar indexli hale getirmek için
    

    # En son bir kez daha sıralayacağımız için  ratinge göre ilk 100 filmi alıyoruz 
    top50 = genreOfMovieStats.sort_values(by='avgOfRating', ascending=False).head(100)

        # Eğer 50 film yoksa, diğer türlerden ekleme yap
    if len(top50) < 50:
        # Eksik sayıyı bul
        missingCount = 50 - len(top50)

        # Diğer türlerden uygun filmleri al
        otherGenres = moviedf[moviedf['mainTag'].str.lower() != genre.lower()]

        # Diğer türlerden en çok izlenen ve beğenilen filmleri seç
        otherGenreStats = otherGenres.groupby('movieId').agg(
            title=('title', 'first'), 
            genres=('genres', 'first'),
            mainTag=('mainTag', 'first'),
            numOfRatings=('numOfRatings', 'first'), 
            avgOfRating=('avgOfRating', 'first')
        ).reset_index()

        # Tür filtresi uygulama
        otherGenreStats = otherGenreStats[otherGenreStats['genres'].str.contains(genre, case=False, na=False)]

        # Ekleyeceğimiz filmleri belirleyelim
        additionalMovies = otherGenreStats.sort_values(by=['numOfRatings', 'avgOfRating'], ascending=[False, False]).head(missingCount)

        # Eklenmesi gereken filmlerin izlenme sayısını ve ortalama derecelerini kontrol edelim
        for _, row in additionalMovies.iterrows():
            # Mevcut filmlerin izlenme sayısı ve ortalama derecesi
            currentRatingsCount = top50['numOfRatings'].mean()
            currentAvgRating = top50['avgOfRating'].mean()

            # Eğer yeni film mevcut filmlerden %10 daha iyi ise ekleyelim
            if (row['numOfRatings'] > currentRatingsCount * 1.1) and (row['avgOfRating'] > currentAvgRating * 1.1):
                top50 = pd.concat([top50, additionalMovies], ignore_index=True)
                break  # Eklemeyi yaptıktan sonra döngüyü kır

    # `numOfRatings` değeri 5000'den büyük olanları yalnızca `avgOfRating`e göre sıralıyoruz
    top_ratings_over_5000 = top50[top50['numOfRatings'] > 5000].sort_values(by='avgOfRating', ascending=False)

    # `numOfRatings` değeri 5000'den küçük olanları `avgOfRating` ve `numOfRatings`e göre sıralıyoruz
    top_ratings_under_5000 = top50[top50['numOfRatings'] <= 5000].sort_values(by=['numOfRatings', 'avgOfRating'], ascending=[False, False])

    # İki tabloyu birleştirip `top50`'ye atıyoruz
    top50 = pd.concat([top_ratings_over_5000, top_ratings_under_5000]).head(50)

    return top50

top50moviesByGenredf = {} # çıktıları saklamak için boş küme

for genre in genres10:
    top50moviesByGenredf[genre] = top50moviesByGenre(genre)  # çıktıyı atama

    # CSV dosya adını oluşturma
    fileName = f"archive/top50moviesByGenres/top50moviesOf{genre}.csv"

    # DataFrame'i CSV dosyasına yazma
    top50moviesByGenredf[genre].to_csv(fileName, index=False)  




# Her tür için CSV dosyasını okuyup değişkenlere kaydetme
for genre in genres10:
    file_name = f"archive/top50moviesByGenres/top50moviesOf{genre}.csv"
    globals()[f"{genre.lower()}df"] = pd.read_csv(file_name)



########################################### Her türdeki en çok izlenen filmleri tek bir csv dosyasında birleştirme


dataframes=[]
for genre in genres10:
    fileName = f"archive/top50moviesByGenres/top50moviesOf{genre}.csv"
    df = pd.read_csv(fileName)
    dataframes.append(df)

#Türler içinde en iyi 50 filmin dataframeinin tek dataframe'de birleştiren fonksiyon
def CombineAndSortByMovieId(dataframes): 
    
    combineddf = pd.concat(dataframes, ignore_index=True)

    deletedVersOfRedundancy = combineddf.groupby('movieId').agg(
        title = ('title','first'),
        genres = ('genres', 'first'),
        numOfRatings = ('numOfRatings', 'first'),
        avgOfRating = ('avgOfRating', 'first') 
    ).reset_index()
    sorteddf = deletedVersOfRedundancy.sort_values(by='movieId')
    
    return sorteddf

moviesTOPdf = CombineAndSortByMovieId(dataframes) 



#Filmler için en iyiler listesini csvye kaydetme
moviesTOPdf.to_csv('archive/moviesTOP.csv', index=False) 


##################################################################      Matrise Çevirme

# İzleyici/film matrisi oluşturma
userMovieMatrix = movieUserdf.pivot(index='userId', columns='movieId', values='rating').notnull().astype(np.int8)
print(userMovieMatrix)

################################################################## Apriori Algoritması



# Kullanıcı-film matrisini bool değerine çeviriyoruz.
userMovieMatrixbool = userMovieMatrix.astype(bool)

# Apriori algoritması ile sık film gruplarını bulma
frequent_itemsets = apriori(userMovieMatrixbool, min_support=0.045, use_colnames=True)

# Association Rules fonksiyonuyla destek, güven ve lift değerleriyle kuralları oluşturma

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)


rulesdf = rules.copy()

# frozenset'leri int'e çevirme
rulesdf['antecedents'] = rulesdf['antecedents'].apply(lambda x: ', '.join(map(str, x)))  
rulesdf['consequents'] = rulesdf['consequents'].apply(lambda x: ', '.join(map(str, x)))   




def OrderingRules(rulesdf, movieId):
    # Seçilen ID'yi string olarak kontrol et
    movieId_str = str(movieId)

    # Seçilen ID'yi içeren satırları filtreleme
    resultdf = rulesdf[rulesdf['antecedents'].apply(lambda x: movieId_str in [id.strip() for id in x.split(',')])].copy()

    # Çarpma işlemini yeni sütun olarak ekle
    resultdf['product'] = resultdf['support'] * resultdf['confidence'] * resultdf['lift']

    # Consequents'i ayırma
    resultdf['consequents'] = resultdf['consequents'].apply(lambda x: [int(i.strip()) for i in x.split(',')])
    resultdf = resultdf[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'product']]
    # İstenilen sütunları döndür
    return resultdf

#rulesdf[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv('archive/rules.csv', index=False)


# Veri yapısı örneği
users = movieUserdf['userId'].unique()  # Veri setinde yer alan kullanıcı ID'leri
genres = genres10  # Film türleri

def truncateString(string, max_length):
    if len(string) > max_length:
        return string[:max_length - 3] + "..." 
    return string


def GetPopularRecommendation(genre=None, movie_id=None):
    suggestIds = []

    if genre is None:
        if movie_id is not None:
            movie_id = int(movie_id)
            
            rulesAs2Id = OrderingRules(rulesdf, movieId=movie_id)
            if not rulesAs2Id.empty:  # rulesAs2Id boş değilse
                rulesAs2Id = rulesAs2Id[rulesAs2Id['support'] > 0.04]  # 0.04'den büyük olanları al
                if not rulesAs2Id.empty:  # Yine kontrol
                    # Consequents içindeki değerleri düzleştirerek suggestIds'e ekle
                    for conseq in rulesAs2Id['consequents']:
                        if isinstance(conseq, list):
                            suggestIds.extend(conseq)  # veya suggestIds += conseq
                        else:
                            suggestIds.append(conseq)
        

            suggestIds = list(set(suggestIds))
            suggestedMovies = moviedf[moviedf['movieId'].isin(suggestIds)]
            if suggestedMovies.empty:
                return f"Üzgünüm! Öneri için film bulunamadı."
            else:
                # Başlıkları ve filmleri biçimlendirerek döndür
                formatted_movies = "\n".join([
                    f"{truncateString(row['title'], 45)}\t\t\t\t\t{row['genres']}\t\t\t\t\t{row['mainTag'].capitalize()}\t\t\t\t\t{round(row['avgOfRating'], 2)}\t\t\t\t\t{row['numOfRatings']}" 
                    for index, row in suggestedMovies.iterrows()
                ])
        else:
            return f"movieId eşleşmedi."
    elif genre is not None:
        genredf = globals()[f"{genre}df"]  # Belirli film türündeki veriye erişim
        
        # Başlıkları ve filmleri biçimlendirerek döndür
        
        formatted_movies = "\n".join([
            f"{truncateString(row['title'], 45)}\t\t\t\t\t{row['genres']}\t\t\t\t\t{row['mainTag'].capitalize()}\t\t\t\t\t{round(row['avgOfRating'], 2)}\t\t\t\t\t{row['numOfRatings']}" 
            for index, row in genredf.iterrows()
        ])
        
    return formatted_movies



def GetPersonalizedRecommendation(user_id, movie_id=None, genre=None):
    user_id = int(user_id)
    watchedMoviesdf = movieUserdf[movieUserdf['userId'] == user_id]
    watchedMoviesdf = watchedMoviesdf.sort_values(by='rating', ascending=False)
    suggestIds = []

    if movie_id is None:
        
        xwatchedMoviesdf = moviedf[moviedf['movieId'].isin(watchedMoviesdf['movieId'])] # moviedf deki veriler daraltıldığı için watchedMoviesdf ile ortak olanları alıyoruz

        for index, movie in xwatchedMoviesdf.iterrows():
            rulesAs2Id = OrderingRules(rulesdf, movieId=movie['movieId'])
            if not rulesAs2Id.empty:  # rulesAs2Id boş değilse
                rulesAs2Id = rulesAs2Id[rulesAs2Id['confidence'] > 0.25]  # 0.25'den büyük olanları al
                if not rulesAs2Id.empty:  # Yine kontrol
                    # Consequents içindeki değerleri düzleştirerek suggestIds'e ekle
                    for conseq in rulesAs2Id['consequents']:
                        if isinstance(conseq, list):
                            suggestIds.extend(conseq)  # veya suggestIds += conseq
                        else:
                            suggestIds.append(conseq)
    elif movie_id is not None:
        movie_id = int(movie_id)
        
        rulesAs2Id = OrderingRules(rulesdf, movieId=movie_id)
        if not rulesAs2Id.empty:  # rulesAs2Id boş değilse
            rulesAs2Id = rulesAs2Id[rulesAs2Id['confidence'] > 0.2]  # 0.2'den büyük olanları al
            if not rulesAs2Id.empty:  # Yine kontrol
                # Consequents içindeki değerleri düzleştirerek suggestIds'e ekle
                for conseq in rulesAs2Id['consequents']:
                    if isinstance(conseq, list):
                        suggestIds.extend(conseq)  # veya suggestIds += conseq
                    else:
                        suggestIds.append(conseq)
    

    suggestIds = list(set(suggestIds))
    suggestIds = [id for id in suggestIds if id not in watchedMoviesdf['movieId']]

    # Eğer genre belirtilmişse, önerilen filmleri genre'ye göre filtrele
    if genre is not None:
        suggestedMovies = moviedf[moviedf['movieId'].isin(suggestIds) & (moviedf['genres'].str.lower().str.contains(genre))]
    elif genre is None:
        suggestedMovies = moviedf[moviedf['movieId'].isin(suggestIds)]

    if suggestedMovies.empty:
        return f"Üzgünüm! Öneri için film bulunamadı."
    else:
        formatted_movies = "\n".join([f"{truncateString(row['title'], 45)}\t\t\t\t\t{row['genres']}\t\t\t\t\t{row['mainTag'].capitalize()}\t\t\t\t\t{round(row['avgOfRating'], 2)}\t\t\t\t\t{row['numOfRatings']}" for index, row in suggestedMovies.iterrows()])
        return formatted_movies
    
def setup_text_styles():
    # Kalın (bold) stil tanımlaması
    output_text.tag_configure("bold", font=("Helvetica", 10, "bold"))
    # Film türünü ve diğer metinleri için normal stil
    output_text.tag_configure("normal", font=("Helvetica", 10))


# Öneri fonksiyonları çağıran ana fonksiyon
def recommend_movie():
    recommendation_type = recommendation_type_var.get()
    condition_type = condition_type_var.get()

    # Text stil ayarlarını yapılandır
    setup_text_styles()

    if recommendation_type == "Popüler Film Önerileri":
        if condition_type == "Film Türüne Göre Öneriler":
            if genre_listbox.curselection():
                genre = genre_listbox.get(genre_listbox.curselection())
                recommendation = GetPopularRecommendation(genre)
            else:
                recommendation = "Lütfen bir tür seçin."
        elif condition_type == "Film İsmine Göre Öneriler":
            if movies_listbox.curselection():
                movie_name = movies_listbox.get(movies_listbox.curselection())
                movie_id = moviedf[moviedf['title'] == movie_name]['movieId'].values[0]
                recommendation = GetPopularRecommendation(genre=None, movie_id=movie_id)
            else:
                recommendation = "Lütfen bir film seçin"

    elif recommendation_type == "Kişiselleştirilmiş Film Önerileri":
        if user_listbox.curselection():
            user_id = user_listbox.get(user_listbox.curselection())
            if condition_type == "Film Türüne Göre Öneriler":
                if genre_listbox.curselection():
                    genre = genre_listbox.get(genre_listbox.curselection())
                    recommendation = GetPersonalizedRecommendation(user_id, movie_id=None, genre=genre)
                else:
                    recommendation = "Lütfen bir tür seçin."
            elif condition_type == "Film İsmine Göre Öneriler":
                if movies_listbox.curselection():
                    movie_name = movies_listbox.get(movies_listbox.curselection())
                    movie_id = moviedf[moviedf['title'] == movie_name]['movieId'].values[0]
                    recommendation = GetPersonalizedRecommendation(user_id, movie_id=movie_id, genre=None)
                else:
                    recommendation = "Lütfen bir film seçin."
            else:
                recommendation = "Lütfen bir film seçin"
        else:
            recommendation = "Lütfen bir kullanıcı seçin."

    output_text.delete(1.0, tk.END)

    # Başlıkları ekleyelim
    header = "Film ismi\t\t\t\t\t\tF. Türleri\t\t\t\tF. Ana Türü\t\t\t\t\tOrt. Rating\t\t\t\t\tİzlenme\n"
    output_text.insert(tk.END, header, "bold")

    # Film bilgilerini formatlayarak ekleyelim
    if isinstance(recommendation, str) and "Lütfen bir" not in recommendation:
        formatted_movies = recommendation.split("\n")
        for movie in formatted_movies:
            output_text.insert(tk.END, movie + "\n", "normal")
    else:
        output_text.insert(tk.END, recommendation, "normal")

# Ana pencere oluşturma
root = tk.Tk()
root.title("Film Öneri Sistemi")
root.geometry("1600x800")


background_color = "#121212"  # Koyu gri
foreground_color = "#FFFFFF"  # Beyaz
highlight_color = "#E50915"   # Kırmızı

root.configure(bg=background_color)

# Çıktı ekranı
output_label = tk.Label(root, text="Önerilen Filmler:", font=("Arial", 14, "bold"), bg=background_color, fg=highlight_color)
output_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
output_text = tk.Text(root, height=12, width=175, wrap="word", font=("Arial", 12), bg="#333333", fg=foreground_color)
output_text.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Öneri türü seçimi için radio button
recommendation_type_var = tk.StringVar(value="Popüler Film Önerileri")
recommendation_type_label = tk.Label(root, text="Öneri Türünü Seçin:", font=("Arial", 12, "bold"), bg=background_color, fg=foreground_color)
recommendation_type_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

# Radio button seçenekleri
popular_radio = tk.Radiobutton(root, text="Popüler Film Önerileri", variable=recommendation_type_var,
                               value="Popüler Film Önerileri", font=("Arial", 12),
                               command=lambda: toggle_widgets(), bg=background_color, fg="SystemButtonFace", selectcolor=highlight_color,
                               activebackground=background_color, activeforeground=foreground_color)
personalized_radio = tk.Radiobutton(root, text="Kişiselleştirilmiş Film Önerileri", variable=recommendation_type_var,
                                    value="Kişiselleştirilmiş Film Önerileri", font=("Arial", 12),
                                    command=lambda: toggle_widgets(), bg=background_color, fg=foreground_color, selectcolor=highlight_color,
                                    activebackground=background_color, activeforeground=foreground_color)
popular_radio.grid(row=2, column=1, sticky="w", padx=10, pady=5)
personalized_radio.grid(row=2, column=2, sticky="w", padx=10, pady=5)

# Kullanıcı seçimi listbox'ı (Kişiselleştirilmiş öneri için)
user_listbox_label = tk.Label(root, text="Kullanıcı Id'si Seçin:", font=("Arial", 12, "bold"), bg=background_color, fg=foreground_color)
user_listbox = tk.Listbox(root, height=4, font=("Arial", 12), bg="#333333", fg=foreground_color, selectbackground=highlight_color, exportselection=False)
for user in users:
    user_listbox.insert(tk.END, user)

user_listbox_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
user_listbox.grid(row=3, column=1, padx=10, pady=10)
user_listbox_label.grid_remove()  # Başlangıçta gizli
user_listbox.grid_remove()  # Başlangıçta gizli

# Öneri kriteri seçimi için radio button
condition_type_var = tk.StringVar(value="Film Türüne Göre Öneriler")
condition_type_label = tk.Label(root, text="Öneri Kriterini Seçin:", font=("Arial", 12, "bold"), bg=background_color, fg=foreground_color)
condition_type_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

genre_radio = tk.Radiobutton(root, text="Film Türüne Göre Öneriler", variable=condition_type_var,
                             value="Film Türüne Göre Öneriler", font=("Arial", 12),
                             command=lambda: toggle_genre(), bg=background_color, fg=foreground_color, selectcolor=highlight_color,
                             activebackground=background_color, activeforeground=foreground_color)
name_radio = tk.Radiobutton(root, text="Film İsmine Göre Öneriler", variable=condition_type_var,
                            value="Film İsmine Göre Öneriler", font=("Arial", 12),
                            command=lambda: toggle_genre(), bg=background_color, fg=foreground_color, selectcolor=highlight_color,
                            activebackground=background_color, activeforeground=foreground_color)
genre_radio.grid(row=4, column=1, sticky="w", padx=10, pady=5)
name_radio.grid(row=4, column=2, sticky="w", padx=10, pady=5)

# Film türüne göre öneri yapılırken tür seçimi listbox'ı
genre_listbox_label = tk.Label(root, text="Film Türünü Seçin:", font=("Arial", 12, "bold"), bg=background_color, fg=foreground_color)
genre_listbox = tk.Listbox(root, height=10, font=("Arial", 12), bg="#333333", fg=foreground_color, selectbackground=highlight_color, exportselection=False)
for genre in genres:
    genre_listbox.insert(tk.END, genre)

genre_listbox_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
genre_listbox.grid(row=5, column=1, padx=10, pady=10)
genre_listbox_label.grid_remove()  # Başlangıçta gizli
genre_listbox.grid_remove()  # Başlangıçta gizli

# Film ismi seçimi listbox'ı
movies_listbox_label = tk.Label(root, text="Film ismini seçin:", font=("Arial", 12, "bold"), bg=background_color, fg=foreground_color)
movies_listbox = tk.Listbox(root, height=10, font=("Arial", 12), bg="#333333", fg=foreground_color, selectbackground=highlight_color, exportselection=False)

def update_movies_listbox():
    # Film listbox'ını güncellemek için önce temizle
    movies_listbox.delete(0, tk.END)  
    # Seçim durumuna göre kullanıcıya uygun filmleri ekle
    if condition_type_var.get() == "Film İsmine Göre Öneriler":
        if recommendation_type_var.get() == "Kişiselleştirilmiş Film Önerileri":
            if user_listbox.curselection():
                user_id = user_listbox.get(user_listbox.curselection())
                user_id = int(user_id)
                watchedMovies = movieUserdf[movieUserdf['userId'] == user_id]

                for movie_id in watchedMovies['movieId']:
                    movie_name = moviedf.loc[moviedf['movieId'] == movie_id, 'title'].values[0]
                    movies_listbox.insert(tk.END, movie_name)
            else:
                movies_listbox.insert(tk.END, "Lütfen bir kullanıcı seçin.")
        elif recommendation_type_var.get() == "Popüler Film Önerileri":
            for movie_name in moviesTOPdf['title']:
                movies_listbox.insert(tk.END, movie_name)

# Seçim değişikliklerini izleyip listbox'ı güncelle
recommendation_type_var.trace_add("write", lambda *args: update_movies_listbox())
condition_type_var.trace_add("write", lambda *args: update_movies_listbox())
user_listbox.bind("<<ListboxSelect>>", lambda e: update_movies_listbox())

# Listbox'ları arayüze yerleştir
movies_listbox_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")
movies_listbox.grid(row=6, column=1, padx=10, pady=10)
movies_listbox_label.grid_remove()  # Başlangıçta gizli
movies_listbox.grid_remove()  # Başlangıçta gizli

# Öneri butonu
recommend_button = tk.Button(root, text="Öneriyi Göster", font=("Arial", 12, "bold"), command=recommend_movie)
recommend_button.grid(row=6, column=0, columnspan=3, pady=20)

# Widgetları gizlemek veya göstermek için fonksiyon
def toggle_widgets():
    if recommendation_type_var.get() == "Kişiselleştirilmiş Film Önerileri":
        user_listbox_label.grid()
        user_listbox.grid()
    else:
        user_listbox_label.grid_remove()
        user_listbox.grid_remove()

# Öneri kriterine göre listbox'ı gösterme/gizleme ve buton pozisyonunu ayarlama fonksiyonu
def toggle_genre():
    if condition_type_var.get() == "Film Türüne Göre Öneriler":
        genre_listbox_label.grid()
        genre_listbox.grid()
        movies_listbox_label.grid_remove()
        movies_listbox.grid_remove()
        recommend_button.grid(row=6, column=0, columnspan=3, pady=20)  # Butonu listbox altına al
    elif condition_type_var.get() == "Film İsmine Göre Öneriler":
        movies_listbox_label.grid()
        movies_listbox.grid()
        genre_listbox_label.grid_remove()
        genre_listbox.grid_remove()
        recommend_button.grid(row=7, column=0, columnspan=3, pady=20)  # Butonu listbox altına al


# Öneri fonksiyonu
def recommend_movie():
    output_text.delete(1.0, tk.END)  # Çıktı ekranını temizle
    recommendation_type = recommendation_type_var.get()
    condition_type = condition_type_var.get()

    try:
        if recommendation_type == "Kişiselleştirilmiş Film Önerileri":
            # Kullanıcı ID'si seçilmiş mi kontrol et
            if user_listbox.curselection():
                selected_user_id = user_listbox.get(user_listbox.curselection())
                
                # Seçime göre film türü veya film ID'sine göre öneri yap
                if condition_type == "Film Türüne Göre Öneriler":
                    if genre_listbox.curselection():
                        selected_genre = genre_listbox.get(genre_listbox.curselection())
                        recommendation = GetPersonalizedRecommendation(selected_user_id, genre=selected_genre)
                    else:
                        recommendation = "Lütfen bir tür seçin."
                
                elif condition_type == "Film İsmine Göre Öneriler":
                    if movies_listbox.curselection():
                        selected_movie_id = movies_listbox.get(movies_listbox.curselection())
                        recommendation = GetPersonalizedRecommendation(selected_user_id, movie_id=selected_movie_id)
                    else:
                        recommendation = "Lütfen bir film seçin."
                else:
                    recommendation = GetPersonalizedRecommendation(selected_user_id)
            else:
                recommendation = "Lütfen bir kullanıcı seçin."
        
        elif recommendation_type == "Popüler Film Önerileri":
            # Popüler öneriler için film türü veya film ID'sine göre seçim yap
            if condition_type == "Film Türüne Göre Öneriler":
                if genre_listbox.curselection():
                    selected_genre = genre_listbox.get(genre_listbox.curselection())
                    recommendation = GetPopularRecommendation(genre=selected_genre)
                else:
                    recommendation = "Lütfen bir tür seçin."
            
            elif condition_type == "Film İsmine Göre Öneriler":
                if movies_listbox.curselection():
                    selected_movie_id = movies_listbox.get(movies_listbox.curselection())
                    recommendation = GetPopularRecommendation(movie_id=selected_movie_id)
                else:
                    recommendation = "Lütfen bir film seçin."
            else:
                recommendation = GetPopularRecommendation()
        
        output_text.insert(tk.END, recommendation)
    
    except Exception as e:
        output_text.insert(tk.END, f"Hata oluştu: {str(e)}")

# Uygulama döngüsü
root.mainloop()