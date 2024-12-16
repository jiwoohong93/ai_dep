
def data_loader_movielens_additional():
    path = './data/MovieLens_1M/'
    num_users, num_items = 6040, 3952

    user, _ = load_users(path)
    genre = load_items(path, option='single')
    item = {}
    item['M'] = genre['War']+genre['Crime']+genre['Film-Noir']+genre['Sci-Fi']
    item['F'] = genre['Children\'s']+genre['Fantasy']+genre['Musical']+genre['Romance']
    
    return user, item

def load_users(path):
    f = open(path + "users.dat")
    lines = f.readlines()

    gender, age = {}, {} # generate dictionaries
    gender_index, age_index = ['M', 'F'], [1, 18, 25, 35, 45, 50, 56]

    for i in gender_index:
        gender[i] = []
    for i in age_index:
        age[i] = []  
    for line in lines:
        user, g, a, *args = line.split("::")
        gender[g].append(int(user) - 1)
        age[int(a)].append(int(user) - 1) 

    return gender, age

def load_items(path, option='multiple_genre'):
    f = open(path + "movies.dat", encoding = "ISO-8859-1")
    lines = f.readlines()
    
    genre={}
    genre_index = ['Action', 'Adventure', 'Animation', 'Children\'s', 
                   'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                   'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    for idx in genre_index:
        genre[idx] = []

    for line in lines:
        item, _, tags = line.split("::")
        tags = tags.split('|')
        tags[-1] = tags[-1][:-1]
        if option=='multiple_genre':
            for tag in tags:
                genre[tag].append(int(item) - 1)
        else:
            tag = tags[0]
            genre[tag].append(int(item)-1)
    return genre