import graphlab as gl




def add_path(base, name):
    return base+name



if __name__ == '__main__':
    base_path = "./data/sample-movie-recommender-master/dataset/ml-20m/"
    ratings_path = add_path(base_path, "ratings.csv")
    movies_path = add_path(base_path, "movies.csv")
    ratings = gl.SFrame.read_csv(ratings_path)
    movies = gl.SFrame.read_csv(movies_path)

    #we can rename columns
    #ratings.rename({"userId":"userid", "movieId":"movieid"})

    
    #first make a train test split
    train, test = gl.recommender.util.random_split_by_user(ratings, 'userId', 'movieId', max_num_users=1000, item_test_proportion=0.2 )

    #might also use
    #train, test = ratings.random_split(0.8, seed=10)

    #graphlab can pick a recommender for you
    #recommender = gl.recommender.create(train, 'userId', 'movieId')

    #but maybe we want to specify a factorization recommender
    recommender = gl.recommender.factorization_recommender.create(test, 'userId', 'movieId', 'rating', max_iterations=5)

    top_movies = recommender.recommend([1])['movieId']
    movies.filter_by(top_movies, 'movieId')

    #movies.rename({'movieId':'movieid'})

    recommender_with_side_data = gl.factorization_recommender.create(test, 'userId', 'movieId', 'rating', item_data=movies, max_iterations=5)

    top_movies = recommender_with_side_data.recommend([1])['movieId']
    movies.filter_by(top_movies, 'movieId')

    #we can get similar movies
    inception_id = movies.filter_by('Inception (2010)', 'title')['movieId']
    similar_movies = recommender.get_similar_items(inception_id)['similar']
    movies.filter_by(similar_movies, 'movieId')

    #we can make a visualization

    data_dir = './data/sample-movie-recommender-master/dataset/ml-20m/'
    urls  = gl.SFrame.read_csv(path.join(data_dir, 'movie_urls.csv'))
    #urls.rename({'movieId':'movieid'})
    movies = movies.join(urls, on='movieId')
    users = gl.SFrame.read_csv(path.join(data_dir, 'user_names.csv'))

    view = recommender.views.overview(validation_set=test, user_data=users, user_name_column='name', item_data=movies, item_name_column='title', item_url_column='url')
    view.show()


