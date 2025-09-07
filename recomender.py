from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize,OneHotEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
# ! term → TF (Term Frequency) → IDF(Inverse Document Frequency) → vectorization → sparse matrix


df = pd.read_parquet('temp_data/movie_link_metadata.parquet')

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
v_overview = tfidf.fit_transform(df["overview"].fillna(""))

v_title = tfidf.fit_transform(df['title_ml'].fillna(""))

count_vect = CountVectorizer(tokenizer=lambda x: x.split('|'))
v_genres_ml = count_vect.fit_transform(df["genres_ml"].fillna(""))

lang_enc = OneHotEncoder(handle_unknown="ignore")
v_lang = lang_enc.fit_transform(df[["original_language"]].fillna(""))  # OneHotEncoder has to feed 2D df into it

scaler = StandardScaler()
v_runtime = scaler.fit_transform(df[["runtime"]].fillna(0))

# X = hstack([v_title, v_overview, v_genres_ml, v_lang, v_runtime])
# print(X.shape) # (27278, 5104)


# --------------------------------------------------------------------------------------------------------------
def search_by_title(query, top_k=10):
    q_vec = tfidf.transform([query])        # 變成 1×F 稀疏向量
    scores = linear_kernel(q_vec, v_title)  # 1×N
    scores = scores.ravel()  # 有點像是把他從原先list unbox
    idx = np.argpartition(scores, -top_k)[-top_k:]       # top_k 個索引（未排序）單純是index
    idx = idx[np.argsort(scores[idx])[::-1]]             # 這邊得到的都是 index 而 mask則會是一條布林的series，這是為什麼這會失效
    return df.iloc[idx][["title_ml", "overview"]].assign(score=scores[idx])   # 這也是為什麼要用 iloc 而不是 loc

# print(search_by_title("spider man"))
# a movie regards a journalist become a superhero after been bit by a spider
#                                                title_ml                                           overview     score
# 11826                               Spider-Man 3 (2007)  The seemingly invincible Spider-Man goes up ag...  0.908868
# 7953                                Spider-Man 2 (2004)  Peter Parker is going through a major identity...  0.899937
# 5252                                  Spider-Man (2002)  After being bitten by a genetically altered sp...  0.898449
# 23057                                     Spider (2007)                                               None  0.748681
# 6098                                      Spider (2002)  A mentally disturbed man takes residence in a ...  0.737615
# 19204                    Amazing Spider-Man, The (2012)  Peter Parker is an outcast high schooler aband...  0.716153
# 23215                   The Amazing Spider-Man 2 (2014)  For Peter Parker, life is busy. Between taking...  0.710447
# 4144                         Along Came a Spider (2001)  When a teacher kidnaps a girl from a prestigio...  0.565628
# 15085  Spider-Man: The Ultimate Villain Showdown (2002)  Spider-Man meets some of his greatest foes inc...  0.561619
# 15703                  Spider Forest (Geomi sup) (2004)  A recently widowed TV producer is drawn to an ...  0.556535

# --------------------------------------------------------------------------------------------------------------
nn = NearestNeighbors(metric="cosine", algorithm="brute")
nn.fit(v_overview)
# df[df["title_ml"].str.contains("Spider[- ]?Man", case=False, na=False)]


def similar_by_title(title, k=10):
    new_title = '[- ]?'.join(title.split(' '))
    idx = df[df["title_ml"].str.contains(new_title, case=False, na=False)].index[0]

    dists, nbrs = nn.kneighbors(v_overview[idx], n_neighbors=k+1)  # 包含自己
    nbrs = nbrs.ravel().tolist()
    nbrs = [i for i in nbrs if i != idx][:k]
    return df.iloc[nbrs][["title_ml", "overview"]]

# print(similar_by_title("Spider Man"))
#                               title_ml                                           overview
# 11826              Spider-Man 3 (2007)  The seemingly invincible Spider-Man goes up ag...
# 23215  The Amazing Spider-Man 2 (2014)  For Peter Parker, life is busy. Between taking...
# 19204   Amazing Spider-Man, The (2012)  Peter Parker is an outcast high schooler aband...
# 2613              Arachnophobia (1990)  A large spider from the jungles of South Ameri...
# 7953               Spider-Man 2 (2004)  Peter Parker is going through a major identity...
# 22595              Fear No Evil (1981)  High school student turns out to be personific...
# 15064                  Kick-Ass (2010)  Dave Lizewski is an unnoticed high school stud...
# 26666               Austin High (2011)  What would happen if a group of high school sl...
# 25236               Cat and Dog (1983)  L.A.P.D. Captain Parker finally is going on Ho...
# 5864       Children's Hour, The (1961)  A troublemaking student at a girl's school acc...

# --------------------------------------------------------------------------------------------------------------
V_overview = normalize(v_overview)
V_genres   = normalize(v_genres_ml)
V_lang     = normalize(v_lang)
V_runtime  = normalize(v_runtime)

w_over, w_gen, w_lang, w_run = 0.6, 0.3, 0.08, 0.02  # giving weight

X = hstack([
    V_overview * w_over,
    V_genres   * w_gen,
    V_lang     * w_lang,
    V_runtime  * w_run
]).tocsr()  # tocsr() 是因為原先的會是 coo_matrix

n_nn = NearestNeighbors(metric="cosine", algorithm="brute")
n_nn.fit(X)

def similar_by_title(title, k=10):
    new_title = '[- ]?'.join(title.split(' '))
    idx = df[df["title_ml"].str.contains(new_title, case=False, na=False)].index[0]

    dists, nbrs = n_nn.kneighbors(X[idx], n_neighbors=k+1)
    nbrs = nbrs.ravel().tolist()
    nbrs = [i for i in nbrs if i != idx][:k]

    return df.iloc[nbrs][["title_ml", "genres_ml", "original_language", "runtime", "overview"]]

print(similar_by_title('Spider Man'))

#                                      title_ml                              genres_ml  ... runtime                                           overview
# 11826                     Spider-Man 3 (2007)  Action|Adventure|Sci-Fi|Thriller|IMAX  ...   139.0  The seemingly invincible Spider-Man goes up ag...
# 19204          Amazing Spider-Man, The (2012)           Action|Adventure|Sci-Fi|IMAX  ...   136.0  Peter Parker is an outcast high schooler aband...
# 23215         The Amazing Spider-Man 2 (2014)                     Action|Sci-Fi|IMAX  ...   142.0  For Peter Parker, life is busy. Between taking...
# 7953                      Spider-Man 2 (2004)           Action|Adventure|Sci-Fi|IMAX  ...   127.0  Peter Parker is going through a major identity...
# 14165            Three Musketeers, The (1933)              Action|Adventure|Thriller  ...   210.0                                               None
# 2765                          Saturn 3 (1980)              Adventure|Sci-Fi|Thriller  ...     NaN                                               None
# 1660                      Time Tracers (1995)                Action|Adventure|Sci-Fi  ...     NaN                                               None
# 5112                           Ffolkes (1979)              Action|Adventure|Thriller  ...     NaN                                               None
# 25087  Street Fighter: Assassin's Fist (2014)        Action|Adventure|Drama|Thriller  ...     NaN                                               None
# 18562                        Chronicle (2012)                 Action|Sci-Fi|Thriller  ...    84.0  Three high school students make an incredible ...