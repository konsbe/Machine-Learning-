class IDATA:
    userId: str
    movieId: str
    rating: str
    date: str

class UniqueElements():
    objects = []
    def __init__(self, data: IDATA):
        for item in data:
            values = item.split(',')
            obj = {
                'userId': values[0],
                'movieId': values[1],
                'rating': values[2],
                'date': values[3]
            }
            self.objects.append(obj)

    def unique_movies(self) -> int:
        unique_movies = set(obj['movieId'] for obj in self.objects)
        # print('Unique movies:', len(list(unique_movies)))
        return len(list(unique_movies));

    def unique_users(self) -> int:
        unique_users = set(obj['userId'] for obj in self.objects)
        # print('Unique users:', len(list(unique_users)))
        return len(list(unique_users));