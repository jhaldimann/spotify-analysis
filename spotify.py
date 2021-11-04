import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Statistics():
    spotify = None

    def __init__(self, spotify):
        self.spotify = spotify

    def get_total_streams(self):
        total = 0
        for index, row in self.spotify.iterrows():
            total += row['Streams']
        print(total)

    def print_histogram(self):
        df = self.spotify.head(100)
        px.histogram(df, x='Streams', nbins=100, title='Histogram of the Streamingnumbers').show()

    def show_top_ten(self):
        print(self.spotify.head(10))

    def print_scatter_diagram(self):
        print("The popularity is not an indication for much streams")
        px.scatter(self.spotify.head(20), x="Artist_popularity", y="Streams", color="Artist",
                   hover_data=['Artist', 'Track Name']).show()

    def print_box_diagram(self):
        print("Show the average of the streams and the top songs")
        px.box(self.spotify.head(100), y="Streams").show()

    def print_facets_plot(self):
        print("Show the facets plots")

        px.scatter(self.spotify.head(100), x="Artist_popularity", y="loudness", color="Artist",
                   hover_data=['Artist', 'Track Name'], facet_col='Artist', facet_col_wrap=10, title="Artists and their loudness").show()

    def print_best_artists(self):
        print("Get the top artist of this year")
        art = []
        for index, row in self.spotify.head(100).iterrows():
            art.append(row["Artist"])

        artist = art[0]
        count = 0
        while count < 10:
            counter = 0
            artist = art[0]
            for i in art:
                curr_frequency = art.count(i)
                if curr_frequency > counter:
                    counter = curr_frequency
                    artist = i
            art = list(filter(lambda x: x != artist, art))
            print(artist)
            count += 1

    def animation(self):
        px.scatter(self.spotify.nsmallest(100, "Rank")
                   .sort_values(by=['Artist'], ascending=False),
                   animation_frame="Artist_popularity",
                   x="Artist_follower",
                   y="Streams",
                   range_x=[0, 70000000],
                   range_y=[1000, 1200000000],
                   size="Streams",
                   hover_data=['Artist', 'Track Name'],
                   size_max=30,
                   color='Artist',
                   title='Animation about the Followers and the Popularity').show()

    def calculate_regression(self):
        spotify_csv = pd.read_csv('spotify.csv')
        # Prepare the x location
        X = spotify_csv.iloc[:, 22].values.reshape(-1, 1)
        # Prepare the y location
        Y = spotify_csv.iloc[:, 21].values.reshape(-1, 1)
        linear_regressor = LinearRegression()
        linear_regressor.fit(X, Y)
        pred = linear_regressor.predict(X)

        plt.scatter(X, Y)
        plt.plot(X, pred, color='red')
        plt.show()


def regression_analysis(spotifyStats):
    spotify_regression = spotifyStats.spotify
    fig = px.scatter(
        spotify_regression,
        x='Artist_follower',
        y='Artist_popularity',
        range_y=[0, 100],
        range_x=[0, 20000000],
        title='Regression variables'
    )

    update_fig_layout(fig, 'Artist_followrt and Artist_popularity of Spotify Top Songs', 'Artist_follower', 'Artist_popularity')

    fig.show()
    X = spotify_regression['Artist_follower']
    Y = spotify_regression['Artist_popularity']

    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    print(model.summary())

    fig = sm.qqplot(model.resid.sort_values(), dist=stats.norm, line='s')
    fig.show()

    print("Shapiro")
    print(stats.shapiro(model.resid))

    print("Normaltest")
    print(stats.normaltest(model.resid))

    spotify_regression['fitted'] = model.predict()
    spotify_regression['residuals'] = model.resid

    fig = px.scatter(
        spotify_regression,
        x='fitted',
        y='residuals',
        height=400,
        width=600,
        trendline='lowess',
        trendline_color_override='#00FF00',
        marginal_y='box'
    )

    update_fig_layout(fig, 'Residuals vs fFitted values', 'Residuals', 'Modelled values')

    fig.show()

    fig = px.scatter(
        spotify_regression,
        x='Artist_follower',
        y='Artist_popularity',
        trendline='ols',
        height=400,
        width=600
    )

    update_fig_layout(fig, 'Artist_follower and Artist_popularity of Spotify Top Songs', 'Artist_follower', 'Artist_popularity')

    fig.show()

    sns.lmplot(x='Artist_follower', y='Artist_popularity', data=spotify_regression, height=5)

    plt.show()


def update_fig_layout(fig, title, xaxis, yaxis):
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, font=dict(
        family="Lucida Sans, monospace",
        size=18,
        color="#7f7f7f"
    ))


if __name__ == '__main__':
    spotifyStats = Statistics(pd.read_excel('spotify.xlsx', usecols="B,D:F,M,V,W"))
    spotifyStats.print_box_diagram()
    spotifyStats.print_histogram()
    spotifyStats.print_facets_plot()
    spotifyStats.print_scatter_diagram()
    spotifyStats.animation()
    spotifyStats.print_best_artists()
    spotifyStats.calculate_regression()
    regression_analysis(spotifyStats)
