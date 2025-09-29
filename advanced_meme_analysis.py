import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

"""
ADVANCED MEME SPREAD ANALYSIS (FIXED)
------------------------------
Analyzes the collected Reddit data to understand meme propagation patterns
Run this BEFORE importing into Gephi for deeper insights
"""

print("📊 Loading collected data...")

# Load the data
df_edges = pd.read_csv('reddit_meme_network_edges.csv')
df_nodes = pd.read_csv('reddit_meme_network_nodes.csv')
df_detailed = pd.read_csv('reddit_interactions_detailed.csv')

print(f"✅ Loaded: {len(df_edges)} edges, {len(df_nodes)} nodes, {len(df_detailed)} interactions\n")

# Clean data - remove NaN users and convert to string
df_edges = df_edges.dropna(subset=['user'])
df_detailed = df_detailed.dropna(subset=['user'])
df_edges['user'] = df_edges['user'].astype(str)
df_detailed['user'] = df_detailed['user'].astype(str)

# Remove deleted users
df_edges = df_edges[df_edges['user'] != 'None']
df_detailed = df_detailed[df_detailed['user'] != 'None']

print(f"🧹 After cleaning: {len(df_edges)} edges, {len(df_detailed)} interactions")

# -------------------------
# 1. IDENTIFY SUPER-SPREADERS
# -------------------------
print("=" * 60)
print("1️⃣  SUPER-SPREADERS (Users who spread memes across communities)")
print("=" * 60)

# Users who post/comment in multiple subreddits
user_subreddit_count = df_detailed.groupby('user')['target_subreddit'].nunique().reset_index()
user_subreddit_count.columns = ['user', 'subreddit_count']
cross_community_users = user_subreddit_count[user_subreddit_count['subreddit_count'] > 1].sort_values('subreddit_count', ascending=False)

print(f"\n🌟 Top 10 Super-Spreaders (engage in multiple subreddits):\n")
for idx, row in cross_community_users.head(10).iterrows():
    username = str(row['user'])[:20]  # Truncate long usernames
    print(f"   {username:<20s} → Active in {int(row['subreddit_count'])} different subreddits")

# Save super-spreaders
cross_community_users.to_csv('super_spreaders.csv', index=False)
print(f"\n📂 Saved super-spreaders to 'super_spreaders.csv'")

# -------------------------
# 2. MEME VELOCITY (How fast memes spread)
# -------------------------
print("\n" + "=" * 60)
print("2️⃣  MEME VELOCITY (Engagement patterns over time)")
print("=" * 60)

df_detailed['timestamp'] = pd.to_datetime(df_detailed['timestamp'])
df_detailed['hour'] = df_detailed['timestamp'].dt.hour

# Engagement by hour
hourly_engagement = df_detailed.groupby('hour').size()

print("\n⏰ Peak Meme Activity Hours:")
top_hours = hourly_engagement.sort_values(ascending=False).head(5)
for hour, count in top_hours.items():
    print(f"   {hour:02d}:00 - {count} interactions")

# Plot
plt.figure(figsize=(12, 5))
hourly_engagement.plot(kind='bar', color='steelblue')
plt.title('Meme Engagement by Hour (UTC)', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Interactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('meme_velocity_by_hour.png', dpi=150)
print("\n📊 Saved chart: 'meme_velocity_by_hour.png'")

# -------------------------
# 3. CROSS-POSTING PATTERNS
# -------------------------
print("\n" + "=" * 60)
print("3️⃣  CROSS-POSTING PATTERNS (Memes appearing in multiple subreddits)")
print("=" * 60)

# Find posts that appear in analysis from same users across subreddits
user_subreddit_matrix = df_detailed.groupby(['user', 'target_subreddit']).size().reset_index(name='count')
users_in_multiple = user_subreddit_matrix.groupby('user')['target_subreddit'].count()
multi_sub_users = users_in_multiple[users_in_multiple > 1]

print(f"\n🔄 {len(multi_sub_users)} users engage across multiple subreddits")
print(f"   This shows memes spreading through these bridge users!")

# Create subreddit-to-subreddit connections (via shared users)
subreddit_connections = []

for user in multi_sub_users.index:
    user_subs = df_detailed[df_detailed['user'] == user]['target_subreddit'].unique()
    
    # Create connections between all pairs of subreddits this user engages with
    for i, sub1 in enumerate(user_subs):
        for sub2 in user_subs[i+1:]:
            subreddit_connections.append({
                'source_subreddit': sub1,
                'target_subreddit': sub2,
                'bridge_user': user
            })

df_sub_connections = pd.DataFrame(subreddit_connections)

if len(df_sub_connections) > 0:
    # Count connections between subreddits
    sub_network = df_sub_connections.groupby(['source_subreddit', 'target_subreddit']).size().reset_index(name='shared_users')
    sub_network = sub_network.sort_values('shared_users', ascending=False)
    
    print("\n🌉 Strongest Subreddit Bridges (most shared users):\n")
    for idx, row in sub_network.head(10).iterrows():
        print(f"   r/{row['source_subreddit']:<15s} ↔ r/{row['target_subreddit']:<15s} : {int(row['shared_users'])} shared users")
    
    # Save subreddit network
    sub_network.to_csv('subreddit_network.csv', index=False)
    print(f"\n📂 Saved subreddit connections to 'subreddit_network.csv'")
    print("   → Import this into Gephi to see how subreddits are connected!")

# -------------------------
# 4. ENGAGEMENT METRICS
# -------------------------
print("\n" + "=" * 60)
print("4️⃣  ENGAGEMENT QUALITY (Comment scores)")
print("=" * 60)

avg_scores = df_detailed.groupby('target_subreddit')['comment_score'].agg(['mean', 'median', 'max'])
avg_scores = avg_scores.sort_values('mean', ascending=False)

print("\n⭐ Subreddit Engagement Quality (average comment scores):\n")
for subreddit, row in avg_scores.iterrows():
    print(f"   r/{subreddit:<20s} - Avg: {row['mean']:.1f}, Max: {int(row['max'])}")

# -------------------------
# 5. NETWORK METRICS (Basic)
# -------------------------
print("\n" + "=" * 60)
print("5️⃣  BASIC NETWORK METRICS")
print("=" * 60)

# Build network - using cleaned data
G = nx.Graph()

for _, row in df_edges.iterrows():
    if pd.notna(row['user']) and str(row['user']) != 'None':
        G.add_edge(str(row['user']), row['target_subreddit'], weight=row['weight'])

print(f"\n📈 Network Statistics:")
print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")

if G.number_of_nodes() > 0:
    print(f"   Average Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"   Network Density: {nx.density(G):.4f}")

    # Connected components
    components = list(nx.connected_components(G))
    print(f"   Connected Components: {len(components)}")
    if components:
        print(f"   Largest Component Size: {len(max(components, key=len))} nodes")

    # Most central nodes (by degree)
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\n🎯 Most Connected Nodes (Degree Centrality):\n")
    for node, degree in top_nodes:
        node_type = 'subreddit' if node.startswith('r/') or node in df_detailed['target_subreddit'].unique() else 'user'
        node_display = str(node)[:30]  # Truncate long names
        print(f"   {node_display:<30s} - {degree} connections ({node_type})")

# -------------------------
# 6. COMMUNITY OVERLAP MATRIX
# -------------------------
print("\n" + "=" * 60)
print("6️⃣  COMMUNITY OVERLAP ANALYSIS")
print("=" * 60)

# Create matrix showing user overlap between subreddits
subreddits = df_detailed['target_subreddit'].unique()
overlap_matrix = pd.DataFrame(0, index=subreddits, columns=subreddits)

for sub1 in subreddits:
    users_sub1 = set(df_detailed[df_detailed['target_subreddit'] == sub1]['user'])
    for sub2 in subreddits:
        users_sub2 = set(df_detailed[df_detailed['target_subreddit'] == sub2]['user'])
        overlap = len(users_sub1.intersection(users_sub2))
        overlap_matrix.loc[sub1, sub2] = overlap

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Shared Users'})
plt.title('Subreddit User Overlap Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Subreddit')
plt.ylabel('Subreddit')
plt.tight_layout()
plt.savefig('subreddit_overlap_heatmap.png', dpi=150)
print("\n🔥 Saved heatmap: 'subreddit_overlap_heatmap.png'")
print("   This shows which subreddit communities share the most users!")

overlap_matrix.to_csv('subreddit_overlap_matrix.csv')
print("📂 Saved matrix data to 'subreddit_overlap_matrix.csv'")

# -------------------------
# 7. SUMMARY REPORT
# -------------------------
print("\n" + "=" * 60)
print("📋 SUMMARY REPORT FOR YOUR PROJECT")
print("=" * 60)

most_active_sub = df_detailed.groupby('target_subreddit').size().idxmax() if len(df_detailed) > 0 else "N/A"
peak_hour = hourly_engagement.idxmax() if len(hourly_engagement) > 0 else "N/A"

print(f"""
KEY FINDINGS:
-------------
1. Network Size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges
2. Super-Spreaders: {len(multi_sub_users)} users engage across multiple communities
3. Cross-Community Bridges: {len(df_sub_connections) if len(df_sub_connections) > 0 else 0} connections between subreddits
4. Most Active Subreddit: r/{most_active_sub}
5. Peak Activity: {peak_hour}:00 UTC
6. Network Density: {nx.density(G):.4f} (how interconnected the network is)

FILES GENERATED:
---------------
✅ reddit_meme_network_edges.csv - Main network (for Gephi)
✅ reddit_meme_network_nodes.csv - Node attributes (for Gephi)
✅ super_spreaders.csv - Users spreading memes across communities
✅ subreddit_network.csv - Subreddit-to-subreddit connections (for Gephi)
✅ subreddit_overlap_matrix.csv - Community overlap data
✅ meme_velocity_by_hour.png - Engagement timeline
✅ subreddit_overlap_heatmap.png - Visual overlap analysis

NEXT STEPS:
----------
1. Import 'reddit_meme_network_edges.csv' into Gephi (user-subreddit network)
2. Import 'subreddit_network.csv' into Gephi (subreddit-subreddit network)
3. Use these insights in your project report/presentation
4. The visualizations show meme spread patterns clearly!
""")

print("\n🎉 Analysis complete! Ready for Gephi visualization!")