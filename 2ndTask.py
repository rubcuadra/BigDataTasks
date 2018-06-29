#+7 921 3676742
from pyspark.sql.functions import *
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from configs import spark, F_TASK
from pyspark.ml.feature import CountVectorizer, RegexTokenizer, StopWordsRemover
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector as MLVector
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.ml.feature import CountVectorizer
from emoji_states import EMOJI_POSITIVE, EMOJI_NEGATIVE
import emoji, regex

#Read Data
profiles = spark.read.parquet(f"{F_TASK}/commonUserProfiles.parquet")
friends = spark.read.parquet(f"{F_TASK}/friends.parquet")
friendsP = spark.read.parquet(f"{F_TASK}/friendsProfiles.parquet")
uGroupsS = spark.read.parquet(f"{F_TASK}/userGroupsSubs.parquet")
followers = spark.read.parquet(f"{F_TASK}/followers.parquet")
followersP = spark.read.parquet(f"{F_TASK}/followerProfiles.parquet")
uWallP = spark.read.parquet(f"{F_TASK}/userWallPosts.parquet")
uWallC = spark.read.parquet(f"{F_TASK}/userWallComments.parquet")
uWallL = spark.read.parquet(f"{F_TASK}/userWallLikes.parquet")
groupsP = spark.read.parquet(f"{F_TASK}/groupsProfiles.parquet")


def E1(): #1) count of comments, posts (all), original posts, reposts and likes made by user
    totalProfiles = profiles\
        .filter(profiles._id.isNotNull())\
        .selectExpr("_id as user")

    countComments = uWallC\
        .groupBy(uWallC.from_id)\
        .count()\
        .selectExpr("from_id as user_c", "count as comments")

    countAllPosts = uWallP\
        .groupBy(uWallP.from_id)\
        .count()\
        .selectExpr("from_id as user_ap", "count as allPosts")

    countOPosts = uWallP\
        .filter( uWallP.is_reposted == False )\
        .groupBy(uWallP.from_id)\
        .count()\
        .selectExpr("from_id as user_op", "count as originalPosts")

    countRPosts = uWallP\
        .filter( uWallP.is_reposted == True )\
        .groupBy(uWallP.from_id)\
        .count()\
        .selectExpr("from_id as user_rp","count as repostedPosts")

    countLikes = uWallL\
        .groupBy(uWallL.likerId)\
        .count()\
        .selectExpr("likerId as user_l", "count as likes")

    join_type = "inner"
    return totalProfiles.join(countComments,   col("user")==countComments.user_c,join_type)\
          .join(countAllPosts,col("user")==countAllPosts.user_ap,join_type)\
          .join(countOPosts,  col("user")==countOPosts.user_op,join_type)\
          .join(countRPosts,  col("user")==countRPosts.user_rp,join_type)\
          .join(countLikes,   col("user")==countLikes.user_l,join_type)\
          .select("user","comments","allPosts","originalPosts","repostedPosts","likes")

def E2(): #2) count of friends, groups, followers
    validP = profiles\
        .filter(profiles._id.isNotNull())\
        .selectExpr("_id as user")
    
    profilesWFriends = friends\
        .filter(friends.profile.isNotNull())\
        .groupBy(friends.profile)\
        .count()\
        .selectExpr("profile as fr_user", "count as friends")

    profilesWGroups = uGroupsS\
        .filter(uGroupsS.user.isNotNull())\
        .groupBy(uGroupsS.user)\
        .count()\
        .selectExpr("user as g_user", "count as groups")\

    profilesWFollowers = followers\
        .filter(followers.profile.isNotNull())\
        .groupBy(followers.profile)\
        .count()\
        .selectExpr("profile as fl_user", "count as followers")
    # print(profilesWGroups)
    # print(profilesWGroups.__class__)
    return validP.join(profilesWGroups,   col("user")==profilesWGroups.g_user,"left_outer")\
          .join(profilesWFollowers,col("user")==profilesWFollowers.fl_user,"left_outer")\
          .join(profilesWFriends,  col("user")==profilesWFriends.fr_user,"left_outer")\
          .select("user","groups","followers","friends")

def E3(): #3) count of videos, audios, photos, gifts
    return friendsP\
           .filter(friendsP.counters.isNotNull())\
           .select( sum(friendsP.counters.audios).alias("audios"),\
                    sum(friendsP.counters.photos).alias("photos"),\
                    sum(friendsP.counters.gifts) .alias("gifts"))
    #followerProfiles can run the same query as friendsProfiles, we don't know if they are repeated, just counting one
    #We use friends because it is a bigger dataset

def E4(): #4) count of "incoming" (made by other users) comments, max and mean "incoming" comments per post
    return uWallP\
           .select(sum("comments.count").alias("totalComments"),\
                max("comments.count").alias("maxCommentsInAPost"),\
                mean("comments.count").alias("averageCommentsPerPost"))

def E5(): #5) count of "incoming" likes, max and mean "incoming" likes per post
    return uWallP\
           .select(sum("likes.count").alias("totalLikes"),\
                max("likes.count").alias("maxLikesInAPost"),\
                mean("likes.count").alias("averageLikesPerPost"))

def E6(): #6) count of geo tagged posts
    return uWallP\
           .filter( uWallP.geo.isNotNull() )\
           .select( count().alias("geoTaggedPosts") )\

def E7(): #7) count of open / closed (e.g. private) groups a user participates in
    #flatMap maps the list of members of each group, this will be our key
    #(user_id,is_group_closed)
    #[(38067060,0), (101873673,0), (376290809,0), (39616221,0)]
    #At the end we'll have ( ((38067060,0),4) ,((38067060,1),1),... )
    #Meaning 38067060 belongs to 4 open groups
    
    #Map After Reduce is for converting
    #( ((38067060, 0), 1), ((101873673, 0), 1), ((376290809, 0), 1) )
    #  (38067060,0,1) ... used by the dataFrame
    return groupsP\
        .filter( groupsP.contacts.isNotNull() )\
        .select("contacts.user_id","is_closed")\
        .rdd\
        .flatMap(lambda row: [(j,row.is_closed) for j in row.user_id] )\
        .map(lambda k: (k,1))\
        .reduceByKey(lambda a,b:a+b)\
        .map(lambda k:( k[0][0],k[0][1],k[1] ))\
        .toDF(["user","isClosedGroup","count"])\

def M1(): #1) count of reposts from subscribed and not-subscribed groups
    reposts_t = uWallP \
        .filter(uWallP.is_reposted) \
        .select('owner_id', 'repost_info.orig_owner_id')\
        .withColumnRenamed("owner_id", "user")

    reposts = reposts_t.filter(reposts_t["orig_owner_id"] < 0)

    user_to_group_sub = uGroupsS\
        .select("user", "group")\
        .groupBy("user")\
        .agg(collect_set("group"))\
        .withColumnRenamed("collect_set(group)", "groups")

    def contains(id, groups):
        if not groups:        return False
        if str(id) in groups: return True
        else:                 return False

    contains_udf = UserDefinedFunction(contains)

    temp = reposts.join(user_to_group_sub, "user","left_outer")

    reposts_from = temp\
        .withColumn("from_subscribed", contains_udf(temp.orig_owner_id, temp.groups))

    reposts_from_subscribed = reposts_from\
        .filter(reposts_from.from_subscribed == 'true')\
        .select('user')\
        .groupBy('user')\
        .count()\
        .withColumnRenamed("count", "from_subscribed")

    reposts_not_from_subscribed = reposts_from \
        .filter(reposts_from['from_subscribed'] == 'false') \
        .select('user')\
        .groupBy("user")\
        .count()\
        .withColumnRenamed("count", "not_from_subscribed")

    reposts_count = reposts_from_subscribed\
        .join(reposts_not_from_subscribed, 'user', "full_outer")\
        .fillna(0)

    return reposts_count

def M2(): #2) count of deleted users in friends and followers
    deleted_friends_profiles = friendsP\
        .filter(friendsP.deactivated == "deleted")\
        .select("id", "deactivated")\
        .withColumnRenamed("id", "follower")

    deleted_follower_profiles = followersP\
        .filter(followersP.deactivated == "deleted")\
        .select("id", "deactivated")\
        .withColumnRenamed("id", "follower")

    deleted_friends = friends\
        .join(deleted_friends_profiles, "follower","inner")\
        .select('profile', 'deactivated')\
        .dropDuplicates()\
        .groupBy('profile')\
        .count()\
        .withColumnRenamed('count', 'deleted_fiends_acc')

    deleted_followers = followers\
        .join(deleted_follower_profiles, "follower","inner")\
        .select("profile", "deactivated")\
        .dropDuplicates()\
        .groupBy("profile")\
        .count()\
        .withColumnRenamed("count", "deleted_followers_acc")

    deleted_count = deleted_friends\
        .join(deleted_followers, "profile","full_outer")\
        .fillna(0)

    return deleted_count

def M3(): #3) aggregate (e.g. count, max, mean) characteristics for comments and likes (separtely) made by (a) friends (b) followers per post
    def contains(id, groups): #Filter
        if not groups:         return False
        if str(id) in groups:  return True
        else:                  return False
    contains_udf = UserDefinedFunction(contains)
    
    user_friends = friends\
        .groupBy("profile")\
        .agg(collect_set("follower").alias("friends"))\
        .select("profile", "friends") #Get set of friends for an user

    comments = uWallC.select("post_owner_id", "from_id", "post_id") #Projection
    likes = uWallL.filter(uWallL.itemType=="post")\
                .selectExpr("ownerId as post_owner_id","likerId as from_id","itemId as post_id")

    #Join users with comments
    post_comment_to_relation = comments\
        .withColumnRenamed("post_owner_id", "profile")\
        .join(user_friends, "profile", "left_outer")\
        .withColumn("is_from_friend", contains_udf(col("from_id"), col("friends")))\
        .select("profile", "is_from_friend", "post_id")\
        .filter(col("is_from_friend") == "true")
    
    #Join users with likes
    post_comment_to_relation = likes\
        .withColumnRenamed("post_owner_id", "profile")\
        .join(user_friends, "profile", "left_outer")\
        .withColumn("is_from_friend", contains_udf(col("from_id"), col("friends")))\
        .select("profile", "is_from_friend", "post_id")\
        .filter(col("is_from_friend") == "true")

    comments_from_friends_per_post = post_comment_to_relation.groupBy("post_id")\
                                        .count()\
                                        .groupBy("post_id")\
                                        .agg(max("count").alias("cMax"), mean("count").alias("cMean"), sum("count").alias("cSum"))\
                                        
    likes_from_friends_per_post    = post_comment_to_relation.groupBy("post_id").count()\
                                        .groupBy("post_id")\
                                        .agg(max("count").alias("lMax"), mean("count").alias("lMean"), sum("count").alias("lSum"))\
                                        
    return comments_from_friends_per_post.join(likes_from_friends_per_post,"post_id")

def M4(): #4) aggregate (e.g. count, max, mean) characteristics for comments and likes (separtely) made by (a) friends and (b) followers per user
    def contains(id, groups): #Filter
        if not groups:         return False
        if str(id) in groups:  return True
        else:                  return False
    contains_udf = UserDefinedFunction(contains)
    
    user_friends = friends\
        .groupBy("profile")\
        .agg(collect_set("follower").alias("friends"))\
        .select("profile", "friends") #Get set of friends for an user

    comments = uWallC.select("post_owner_id", "from_id", "post_id") #Projection

    #Join comments with user-SetOfFriends
    post_comment_to_relation = comments\
        .withColumnRenamed("post_owner_id", "profile")\
        .join(user_friends, "profile", "left_outer")\
        .withColumn("is_from_friend", contains_udf(col("from_id"), col("friends")))\
        .select("profile", "is_from_friend", "post_id")\
        .filter(col("is_from_friend") == "true")

    comments_from_friends_per_post = post_comment_to_relation.groupBy("post_id").count()

    aggregates_results = post_comment_to_relation\
        .select("profile", "post_id")\
        .join(comments_from_friends_per_post, "post_id")\
        .groupBy("profile")\
        .agg(max("count"), mean("count"), sum("count"))\
        .sort(desc("sum(count)"))

    return aggregates_results

def M5(): #5) find emoji (separately, count of: all, negative, positive, others) in (a) user's posts (b) user's comments  
    # Returns (post_id, emoji) or (comment_id, post_id, emoji) or (*whatever_keys, emoji)
    def stripEmojis(row,keys):
        body,_id    = row['text'], row['id']
        emoji_list  = []
        data = regex.findall(r'\X', body)  #TODO: PreCompile 
        for word in data:
            if any(char in emoji.UNICODE_EMOJI for char in word):
                emoji_list.append(word)
        # flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', body) 
        # emoji_list += flags
        tp = [row[k] for k in keys] #Create Keys,Emoji
        return [ (*tp,_emj) for _emj in emoji_list ]  

    #Returns Key, this result must go ( ( commentId, postId, emojiType ) , 3)
    def categorizeEmoji(row,keys):
        tp = [row[k] for k in keys]
        if row.emoji in EMOJI_POSITIVE:   toRet = (*tp,"p") #Positive
        elif row.emoji in EMOJI_NEGATIVE: toRet = (*tp,"n") #Negative
        else:                             toRet = (*tp,"o") #OTHER
        return (toRet,row.emoji_count) 

    emojis_in_post_body = uWallP\
                        .filter( uWallP.text != "" )\
                        .select("id","text")\
                        .rdd\
                        .flatMap( lambda row: stripEmojis(row,["id"]) )\
                        .map(lambda k: (k,1))\
                        .reduceByKey(lambda a,b:a+b)\
                        .map(lambda k:( k[0][0],k[0][1],k[1] ))\
                        .toDF(["post_id","emoji","emoji_count"])
                        
    emojis_in_comments = uWallC.filter( uWallC.id.isNotNull() & uWallC.post_id.isNotNull() )\
                                .select("id","post_id","text")\
                                .rdd\
                                .flatMap( lambda row: stripEmojis(row,["id","post_id"]) )\
                                .map(lambda k: (k,1))\
                                .reduceByKey(lambda a,b:a+b)\
                                .map(lambda k:( k[0][0],k[0][1],k[0][2],k[1] ))\
                                .toDF(["comment_id","cpost_id","emoji","emoji_count"])
    
    #CATEGORIZE EMOJIS in P,N,O ; post_id | emoji_type | amount
    categorized_in_post_body = emojis_in_post_body.rdd\
                                .map( lambda row: categorizeEmoji(row,["post_id"]) )\
                                .reduceByKey(lambda a,b:a+b)\
                                .map(lambda k:( *k[0],k[1] ))\
                                .toDF(["post_id","emoji_type","e_type_count"])\
                                .groupBy( "post_id","emoji_type" )\
                                .agg( sum("e_type_count").alias("countInPostBody"))\
                                .select("post_id","emoji_type","countInPostBody")\
                                .fillna(0)
    # return categorized_in_post_body              #post_id|emojisinpost|emojisInAllComments
    categorized_in_comments = emojis_in_comments.rdd\
                                .map( lambda row: categorizeEmoji(row,["comment_id","cpost_id"]) )\
                                .reduceByKey(lambda a,b:a+b)\
                                .map(lambda k:( *k[0],k[1] ))\
                                .toDF(["comment_id","cpost_id","emoji_type","e_type_count"])\
                                .groupBy( "cpost_id","emoji_type" )\
                                .agg( sum("e_type_count").alias("countInAllComments"))\
                                .select("cpost_id","emoji_type","countInAllComments")\
                                .fillna(0)

    #Count ALL EMOJIS, We need a regex or something to differentiate the good emojis from the bad from the etc,etc
    post_totalEmojis_in_comments = emojis_in_comments\
                                    .groupBy(emojis_in_comments.cpost_id)\
                                    .agg(sum("emoji_count").alias("emojisInAllComments"))

    post_totalEmojis_in_post     = emojis_in_post_body\
                                    .groupBy(emojis_in_post_body.post_id)\
                                    .agg(sum("emoji_count").alias("emojisInPost"))\
                                    
    post_emojis_count = post_totalEmojis_in_post.join(post_totalEmojis_in_comments,
                                    post_totalEmojis_in_post.post_id==post_totalEmojis_in_comments.cpost_id,"left_outer")\
                        .select("post_id","emojisInPost","emojisInAllComments")\
                        .fillna(0)
    
    # emojis_in_post_body.show()          #post_id|emoji|emoji_count
    # emojis_in_comments.show()           #comment_id|cpost_id|emoji|emoji_count
    
    # categorized_in_post_body.show()     #post_id|emoji_type|countInPostBody
    # categorized_in_comments.show()      #post_id|emoji_type|emojisInAllComments                 
    return post_emojis_count              #post_id|emojisinpost|emojisInAllComments
    
def A1(): #1) apply LDA and find topics in user's posts (including reposts)
    textToWords = RegexTokenizer(inputCol="text", outputCol="splitted", pattern="[\\P{L}]+" ) #Remove signs and split by spaces
    stopRemover = StopWordsRemover(inputCol="splitted", outputCol="words", 
        stopWords=StopWordsRemover.loadDefaultStopWords("russian")+StopWordsRemover.loadDefaultStopWords("english"))
    countVectorizer = CountVectorizer(inputCol="words", outputCol="features")
    
    #Filter if post id exists?
    data = uWallP\
        .filter( uWallP.text != "" )\
        .select("id","text")\
        .limit(10)\
    
    pipeline = Pipeline(stages=[textToWords, stopRemover, countVectorizer])
    model  = pipeline.fit(data)
    result = model.transform(data)
    corpus = result.select("id", "features").rdd.map(lambda r: [r.id,Vectors.fromML(r.features)]).cache()

    # Cluster the documents into k topics using LDA
    ldaModel = LDA.train(corpus, k=8,maxIterations=100,optimizer='online')
    topics = ldaModel.topicsMatrix()
    vocabArray = model.stages[2].vocabulary #CountVectorizer
    wordNumbers = 20  # number of words per topic
    topicIndices = spark.sparkContext.parallelize(ldaModel.describeTopics(maxTermsPerTopic = wordNumbers))

    def topic_render(topic):  # specify vector id of words to actual words
        terms = topic[0]
        result = []
        for i in range(wordNumbers):
            term = vocabArray[terms[i]]
            result.append(term)
        return result

    topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

    for topic in range(len(topics_final)):
        print ("Topic" + str(topic) + ":")
        for term in topics_final[topic]:
            print (term)
        print ('\n')

    # print( f"Learned Topics (as distrib over vocab of {ldaModel.vocabSize()} words:" )
    # topics = ldaModel.topicsMatrix()
    # for topic in range(k):
    #   print(f"Topic {topic} :")
    #   for word in range( ldaModel.vocabSize() ):
    #       print(f"{topics[word][topic]}")

if __name__ == '__main__':
    results_path = "results/"
    #EASY
    # E1().repartition(1).write.csv(f'{results_path}E1.csv') #E1().show() 
    # E2().repartition(1).write.csv(f'{results_path}E2.csv') #E2().show()
    # E3().repartition(1).write.csv(f'{results_path}E3.csv') #E3().show()
    # E4().repartition(1).write.csv(f'{results_path}E4.csv') #E4().show()
    # E5().repartition(1).write.csv(f'{results_path}E5.csv') #E5().show()
    # E6().repartition(1).write.csv(f'{results_path}E6.csv') #E6().show()
    # E7().repartition(1).write.csv(f'{results_path}E7.csv') #E7().show()
    #MEDIUM
    # M1().repartition(1).write.csv(f'{results_path}M1.csv')
    # M2().repartition(1).write.csv(f'{results_path}M2.csv')
    # M3().repartition(1).write.csv(f'{results_path}M3.csv')
    # M4().repartition(1).write.csv(f'{results_path}M4.csv')
    # M5().repartition(1).write.csv(f'{results_path}M5.csv')
    #ADVANCED
    A1()
