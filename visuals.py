import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os




def save_time_series(series, outpath):
    plt.figure(figsize=(8,3))
    series.plot()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()




def save_bar(x, y, outpath, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(8,4))
    plt.bar(x, y)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()




def save_wordcloud(text, outpath):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()