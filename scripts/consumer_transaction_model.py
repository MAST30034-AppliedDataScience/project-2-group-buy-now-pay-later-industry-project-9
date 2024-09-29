import matplotlib.pyplot as plt
import seaborn as sns



def feature_visualisation(df_pandas, plots):
    """ 
    Create a list of plots for feature visualisation, including bar charts, scatter plot and histogram.
    """
    # Set up a big plot grid
    fig, axes = plt.subplots(6, 3, figsize=(20, 20))  # 4x3 grid of subplots
    fig.tight_layout(pad=5.0)  # Adjust spacing between plots

    for i, (plot_title, (feature, plot_type)) in enumerate(plots.items()):
        ax = axes[i // 3, i % 3]  # Select subplot position

        if plot_type == "hist":
            sns.histplot(df_pandas[feature], bins=30, kde=True, ax=ax)
        elif plot_type == "count":
            sns.countplot(x=feature, data=df_pandas, ax=ax)
        elif plot_type.startswith("scatter"):
            if plot_type == "scatter1":
                sns.scatterplot(x="Proportion_between_max_order_value_mean_income", y="fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter2":
                sns.scatterplot(x="Proportion_between_max_order_value_median_income", y="fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter3":
                sns.scatterplot(x="Proportion_between_total_order_value_mean_income", y="fraud_probability", data=df_pandas, ax=ax)
            elif plot_type == "scatter4":
                sns.scatterplot(x="Proportion_between_total_order_value_median_income", y="fraud_probability", data=df_pandas, ax=ax)

        ax.set_title(plot_title)

    # Show the entire plot
    plt.show()
