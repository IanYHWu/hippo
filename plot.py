import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_from_csv(path, col_x, col_y):
    df = pd.read_csv(path)
    sns.scatterplot(x=col_x, y=col_y, data=df)
    plt.show()


if __name__ == "__main__":
    plot_from_csv('../../first_run_caveflyer.csv', col_x='timesteps', col_y='mean_episode_rewards')

