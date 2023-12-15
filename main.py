import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
zf = zipfile.ZipFile('archive.zip')
df = pd.read_csv(zf.open('spotify_data.csv'), index_col=0)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
df.drop_duplicates(keep='first')
df = df[df['tempo'] != 0]
df = df[df['valence'] != 0]
df = df[df['duration_ms'] != 0]
df = df.loc[df['danceability'] < df['danceability'].quantile(0.99)]
df = df.loc[df['energy'] < df['energy'].quantile(0.99)]
df = df.loc[df['loudness'] < df['loudness'].quantile(0.99)]
df = df.loc[df['speechiness'] < df['speechiness'].quantile(0.99)]
df = df.loc[df['instrumentalness'] < df['instrumentalness'].quantile(0.99)]
df = df.loc[df['acousticness'] < df['acousticness'].quantile(0.99)]
df = df.loc[df['liveness'] < df['liveness'].quantile(0.99)]
df = df.loc[df['valence'] < df['valence'].quantile(0.99)]
df = df.loc[df['tempo'] < df['tempo'].quantile(0.99)]
df = df.loc[df['duration_ms'] < df['duration_ms'].quantile(0.99)]
colsToEncode = ['key','mode','time_signature']
colsToNotEncode = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
corrMatdf = df.drop(columns=['genre','artist_name','track_name','track_id','year'],axis=1)
corrMatdf = pd.get_dummies(corrMatdf, columns=colsToEncode)
correlation_matrix = corrMatdf.corr()

plt.figure(figsize=(10, 6))
plt.hist(df['duration_ms'] / 60000, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Продолжительность трека (в минутах)')
plt.ylabel('Количество треков')
plt.title('Распределение времени треков')
plt.show()

# Ящик с усами популярности по жанрам
plt.figure(figsize=(14, 8))
sns.boxplot(x='genre', y='popularity', data=df, palette='viridis')
plt.xlabel('Жанр')
plt.ylabel('Популярность трека')
plt.title('Популярность треков по жанрам')
plt.xticks(rotation=45, ha='right')
plt.show()

# Распределение популярности в зависимости от музыкальных ключей
plt.figure(figsize=(12, 6))
sns.barplot(x='key', y='popularity', data=df, palette='muted')
plt.xlabel('Музыкальный ключ')
plt.ylabel('Средняя популярность треков')
plt.title('Распределение популярности в зависимости от музыкальных ключей')
plt.show()

instrumental_count = df[df['instrumentalness'] > 0.5].shape[0]
vocal_count = df[df['instrumentalness'] <= 0.5].shape[0]
labels = ['Инструментальные', 'Вокальные']
sizes = [instrumental_count, vocal_count]
colors = ['lightcoral', 'lightskyblue']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Доля инструментальных треков по сравнению с вокальными')
plt.show()

# Распределение популярности треков
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='popularity', bins=20, kde=True)
plt.title('Распределение популярности треков')
plt.xlabel('Популярность')
plt.ylabel('Частота')
plt.show()

# Boxplot для энергии по каждому музыкальному ключу
plt.figure(figsize=(12, 8))
sns.boxplot(x='key', y='energy', data=df)
plt.title('Boxplot для энергии по каждому музыкальному ключу')
plt.xlabel('Музыкальный ключ')
plt.ylabel('Энергия')
plt.show()
# Распределение популярности треков в зависимости от их длительности
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration_ms', y='popularity', data=df, alpha=0.5)
plt.title('Распределение популярности  треков в зависимости от их длительности')
plt.xlabel('Длительность (мс)')
plt.ylabel('Популярность')
plt.show()

# Круговая диаграмма для доли треков по каждому музыкальному ключу
key_counts = df['key'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(key_counts, labels=key_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Доля треков по каждому музыкальному ключу')
plt.show()

