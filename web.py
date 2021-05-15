from classes import *
import streamlit as st
import pickle

# Load Data Function
@st.cache(allow_output_mutation = True)
def load_data():
	anime_df = pd.read_csv('anime.csv')
	
	#Load Model
	model = NCF(353405, 48493, 0, 0)
	model.load_state_dict(torch.load("recommender_model.pt"))
	model.eval()

	#Load All Anime Ids
	with open('all_animeIds.pickle', 'rb') as handle:
	    all_animeIds = pickle.load(handle)
	#Load User Dictionary
	with open('user_set_dict.pickle', 'rb') as handle:
	    user_set_dict = pickle.load(handle)
	#Load Anime Genres
	with open('anime_genres.pickle', 'rb') as handle:
   		anime_genres = pickle.load(handle)
	anime_genres.insert(0, 'All')
	return model, anime_df, anime_genres, user_set_dict, all_animeIds

model, anime_df, anime_genres, user_set_dict, all_animeIds = load_data()
temp = sorted(anime_df['Name'].tolist(), key = str.lower)
anime_list = temp.copy()
st.title("Anime Recommendation System")

anime_watched = st.multiselect("Animes you watched", anime_list)

num_recommend = st.sidebar.selectbox("Number of Recommended Anime (10 ~ 100)", range(10, 101))
genre_recommend = st.sidebar.selectbox("Genre of Recommended Anime", anime_genres)
randomized_recommend = st.sidebar.selectbox("Recommending in comparison to:", ("All Anime", "Randomized Subset"))

print_rec = st.button('Give Recommendations')

if(print_rec):
	closest_user = find_closest_user(anime_watched, user_set_dict)
	random = True
	if(randomized_recommend == "All Anime"):
		random = False

	recAnime = recommend_anime(model, 0, all_animeIds, anime_df, genre_recommend, anime_watched, num_recommend, random)
	num = 1
	for anime in recAnime:
		st.write(str(num) + '. ' + anime)
		num += 1

