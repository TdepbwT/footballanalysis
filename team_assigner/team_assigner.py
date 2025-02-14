from sklearn.cluster import KMeans
class TeamAssigner: # get player color and use to assign team colors
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_cluster_model(self,image):
        # reshape image to 2d array
        image_2d = image.reshape((-1, 3))

        #kmeans clustering
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(image_2d)

        return kmeans




    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        kmeans = self.get_cluster_model(top_half_image)

        # get labels
        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        #get player cluster
        corner_cluster = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_cluster),key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for player_id, player_detections in player_detections.items():
            bbox = player_detections["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
            
            if player_id == 87:
                self.team_colors[1] = player_color

        kmeans = KMeans(n_clusters = 2, init="k-means++", n_init= 1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans # save kmeans model for future use

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        if player_id == 87:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id