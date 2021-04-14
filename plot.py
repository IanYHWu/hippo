import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_from_csv(path, col_x, col_y, smooth=False):
    df = pd.read_csv(path)
    if not smooth:
        sns.scatterplot(x=col_x, y=col_y, data=df)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.show()
    else:
        exp_avg = df[col_y].ewm(com=0.7).mean()
        df['exp'] = exp_avg
        sns.lineplot(x=col_x, y='exp', data=df)
        plt.xlabel(col_x)
        plt.ylabel(col_y + '_smoothed')
        plt.show()


if __name__ == "__main__":
    plot_from_csv('../updated_maze.csv', col_x='timesteps', col_y='mean_episode_rewards', smooth=True)



