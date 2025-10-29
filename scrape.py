import praw
import pandas as pd
from datetime import datetime

# -------------------------
# 1. SETUP REDDIT API
# -------------------------
reddit = praw.Reddit(
    client_id='8rANnN1rLRBLiuwyB1nfhA',          # Get from reddit.com/prefs/apps
    client_secret='qQIcQwUCZPaXIOqWvqwaAxmJ-yGdBg',   # Get from reddit.com/prefs/apps
    user_agent='meme_spread_research_v1.0'
)

print("✅ Connected to Reddit API")

# -------------------------
# 2. TARGET MEME SUBREDDITS
# -------------------------
meme_subreddits = [
    'memes',
    'dankmemes', 
    'wholesomememes',
    'AdviceAnimals',
    'MemeEconomy'
]

# -------------------------
# 3. COLLECT MEME INTERACTIONS
# -------------------------

all_interactions = []

for subreddit_name in meme_subreddits:
    print(f"\n📊 Analyzing r/{subreddit_name}...")
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Get hot posts
        post_count = 0
        for post in subreddit.hot(limit=20):  # Top 20 hot posts
            
            print(f"  📷 Post: {post.title[:50]}...")
            
            # Record post data
            post_data = {
                'post_id': post.id,
                'subreddit': subreddit_name,
                'title': post.title,
                'author': str(post.author),
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'url': post.url
            }
            
            # Get top commenters (these are the engagers)
            post.comments.replace_more(limit=0)  # Remove "load more comments"
            
            for comment in post.comments[:30]:  # Top 30 comments
                all_interactions.append({
                    'user': str(comment.author),
                    'target_subreddit': subreddit_name,
                    'interaction_type': 'comment',
                    'post_id': post.id,
                    'post_title': post.title,
                    'comment_score': comment.score,
                    'timestamp': datetime.fromtimestamp(comment.created_utc)
                })
            
            # Record the post author as well
            all_interactions.append({
                'user': str(post.author),
                'target_subreddit': subreddit_name,
                'interaction_type': 'post',
                'post_id': post.id,
                'post_title': post.title,
                'comment_score': post.score,
                'timestamp': datetime.fromtimestamp(post.created_utc)
            })
            
            post_count += 1
        
        print(f"  ✅ Collected {post_count} posts from r/{subreddit_name}")
        
    except Exception as e:
        print(f"  ❌ Error with r/{subreddit_name}: {e}")

print(f"\n✅ Total interactions collected: {len(all_interactions)}")

# -------------------------
# 4. BUILD NETWORK
# -------------------------

df_interactions = pd.DataFrame(all_interactions)

if len(df_interactions) > 0:
    # Create edge list: user -> subreddit
    edge_list = df_interactions.groupby(['user', 'target_subreddit']).size().reset_index(name='weight')
    
    # Create nodes
    users = df_interactions['user'].unique()
    subreddits = df_interactions['target_subreddit'].unique()
    
    nodes_data = []
    for user in users:
        nodes_data.append({'id': user, 'type': 'user'})
    for sub in subreddits:
        nodes_data.append({'id': sub, 'type': 'subreddit'})
    
    df_nodes = pd.DataFrame(nodes_data)
    
    # Save files
    edge_list.to_csv('reddit_meme_network_edges.csv', index=False)
    df_nodes.to_csv('reddit_meme_network_nodes.csv', index=False)
    df_interactions.to_csv('reddit_interactions_detailed.csv', index=False)
    
    print("\n📂 Files saved:")
    print(f"   - reddit_meme_network_edges.csv ({len(edge_list)} edges)")
    print(f"   - reddit_meme_network_nodes.csv ({len(df_nodes)} nodes)")
    print(f"   - reddit_interactions_detailed.csv (full data)")
    
    print("\n🎉 Ready for Gephi! Import 'reddit_meme_network_edges.csv'")
    
    # Network stats
    print("\n📊 Network Statistics:")
    print(f"   Users who engage with multiple subreddits: {len(edge_list[edge_list['weight'] > 1])}")
    print(f"   Most active subreddit: r/{edge_list.groupby('target_subreddit')['weight'].sum().idxmax()}")
    print(f"   Most active user: {edge_list.groupby('user')['weight'].sum().idxmax()}")

else:
    print("⚠️ No data collected. Check your Reddit API credentials!")
    print("\nGet credentials from: https://www.reddit.com/prefs/apps")
    print("1. Click 'Create App'")
    print("2. Choose 'script'")
    print("3. Copy client_id and client_secret")