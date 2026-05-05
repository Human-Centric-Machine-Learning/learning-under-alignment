# Imports
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

# Matplotlib configuration
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry,bm}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 50,
    "figure.figsize":(20,8),
    "lines.markersize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False
})


def compute_regret_df(config_file="configs/human_alignment.json"):
    """Compute the regret dataframe for all algorithms and datasets based on the results stored in the output directory specified in the configuration file."""
    # parse experiment configuration 
    with open(config_file, 'r') as f:
        config = json.load(f)

    output_root_dir = config["output_dir"]
    algorithms = config["algorithms"]
    datasets = config["datasets"]

    # parse the results into a single dataframe for plotting
    result_dfs = defaultdict(dict)
    optimal_dfs = defaultdict(dict)
    for algorithm in algorithms:
        for dataset in datasets:
            result_file = f"{output_root_dir}/{config['num_iterations']}iters/{dataset.split('.')[0]}/{algorithm}/results.json"
            with open(result_file, 'r') as f:
                results = json.load(f)  
            df = pd.DataFrame(results)
            df["dataset"] = dataset.split('.')[0]   
            result_dfs[algorithm][dataset.split('.')[0]] = df.copy()
            if algorithm == "optimal":
                optimal_df = df.copy()  # Store optimal rewards for later use
                for algo in algorithms:
                    optimal_dfs[algo][dataset.split('.')[0]] = optimal_df
    for algorithm in algorithms:
        for dataset in datasets:
            ds_name = dataset.split('.')[0]
            # compute the expected cumulative regret
            result_dfs[algorithm][ds_name]['regret'] = optimal_dfs[algorithm][ds_name]['reward']- result_dfs[algorithm][ds_name]['reward']

            
    regret_dfs_flattened = []
    for v in result_dfs.values():
        for val in v.values():
            regret_dfs_flattened.append(val)
            
    regret_df = pd.concat(regret_dfs_flattened, ignore_index=True)
    regret_df = regret_df[regret_df['algorithm'] != 'optimal']

    return regret_df, config


algorithm_mapping = {
    "vanilla_contextual": "Algorithm 2 (vanilla baseline)", 
    "algorithm_one": "Algorithm 1 (ours)",
}
dataset_mapping = {
    "group_A": "Group A",
    "group_B": "Group B",
    "group_C": "Group C",
    "group_BP": "Group BP",
    "data_art": "Art",
    "data_census": "Census",
    "data_cities": "Cities",
    "data_sarcasm": "Sarcasm"

}
human_conf_levels = {
    "very low": "Very low",
    "low": "Low",
    "high": "High",
    "very high": "Very high"

}
mapping = {
    "algorithm": algorithm_mapping,
    "dataset": dataset_mapping
}


def p_y_given_b_h_b(data_path, config, fig_path):
    """Plot P(Y=1 | H=h, B=b) for each human confidence level h and AI confidence level b based on the dataset and the configuration."""
    data = pd.read_csv(data_path, dtype=config['data_types'])
    
    # df to later compute P(Y=1 | H=h, B=b) for each (h, b) pair
    human_col = config['column_label_mapping']['human_conf_label']
    machine_col = config['column_label_mapping']['AI_conf_label']
    outcome_prob = config['column_label_mapping']['outcome_prob'] if config['use_outcome_prob'] else config['column_label_mapping']['outcome']
    if config['use_outcome_prob']:
        # convert the outcome_prob column to be between 0 and 1 if it's between 0 and 100
        if data[outcome_prob].max() > 1:
            data[outcome_prob] = data[outcome_prob] / 100
        if data[machine_col].max() > 1:
            data[machine_col] = data[machine_col] / 100
        p_1_given_h_b = (data[[human_col, machine_col, outcome_prob]])
    else:        
        p_1_given_h_b = data[[human_col, machine_col, outcome_prob]].astype({human_col: 'category', machine_col: 'float', outcome_prob: 'float'})
    

    _ = sns.lineplot(data=p_1_given_h_b, x=machine_col, y=outcome_prob, hue=human_col, markers=True, errorbar=('ci', 95), estimator=np.nanmean, palette=sns.color_palette("colorblind", n_colors=data[human_col].nunique()), style=human_col, dashes=False, hue_order=config['human_conf_order'])
    
    plt.yticks(np.arange(0.5, 1.1, 0.5))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(r'AI confidence $b$')
    plt.ylabel(r'$P(Y=1\,|\,H=h,B=b)$')

    # remove the legend completely     
    plt.legend().remove()
    
    # save the figure in the specified path
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f"{fig_path}/p_y_h_b.pdf", bbox_inches='tight')
    plt.clf()


def regret(regret_df, avg_regret_path='', fig_path='', hue="algorithm", add_legend=False, config=None, ds_name=None):
    """Plot the average regret over time for each algorithm based on the regret dataframe, and save the figure in the specified path."""
    if not os.path.exists(f"{avg_regret_path}/avg_regret.csv"):
        os.makedirs(avg_regret_path, exist_ok=True)
        data = regret_df[['iteration', 'timestep', 'regret', hue]].copy()
        
        # Compute a running average of the regret for each 'hue' value 
        data['regret_avg'] = data.groupby([hue, 'iteration'])['regret'].cumsum()

        data.to_csv(os.path.join(avg_regret_path, 'avg_regret.csv'), index=False)
    else:
        data = pd.read_csv(f"{avg_regret_path}/avg_regret.csv")

    # Map algorithm names to more readable labels
    data[hue] = data[hue].map(mapping[hue]) 
    data.rename_axis(columns={hue: hue.capitalize()}, inplace=True)
    ax = sns.lineplot(data=data, x='timestep', y='regret_avg', hue=hue, style=hue, palette=sns.color_palette("colorblind", n_colors=len(mapping[hue])))
         
    # plot no legend and then add the legend outside of the plot
    plt.xlabel(r'$t$')  
    plt.ylabel(r'Empirical Avg. Regret')    
    
    ax.get_legend().remove()
        
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f"{fig_path}/avg_regret.pdf", bbox_inches='tight')

    plt.clf()

def alignment_legend():
    """Plot the legend for the alignment figure, which shows the different human confidence levels with different colors and markers, and save the legend in a separate file."""
    # create figure twice as large as normal to fit the legend and save it in a file alone 
    plt.figure(figsize=(40, 8))
    # create only the legend for the algorithms with the same colors as in the line plots
    handles = [plt.Line2D([0], [0], color=sns.color_palette("colorblind", n_colors=len(human_conf_levels))[i], lw=4) for i in range(len(human_conf_levels))]
    labels = [human_conf_levels[conf] for conf in human_conf_levels]
    
    # set up the handle styles with markers for each human confidence level
    markers = ['+', 's', 'x', 'o']
    for i in range(len(handles)):
        handles[i].set_marker(markers[i])
        handles[i].set_markersize(40)
        handles[i].set_markeredgewidth(15)
        
    
    # do not include axes in the figure and save only the legend
    plt.gca().set_axis_off()
    plt.legend(title=r'Human confidence $h$', handles=handles, labels=labels, ncol=4, loc='center', fontsize=80, frameon=True, title_fontsize=80)
    plt.savefig(f"figures/legend/alignment_legend.pdf", bbox_inches='tight')
    plt.clf()

def regret_legend():
    """Plot the legend for the regret figure, which shows the different algorithms with different colors and line styles, and save the legend in a separate file."""
    # create figure twice as large as normal to fit the legend and save it in a file alone 
    plt.figure(figsize=(40, 8))
    # create only the legend for the algorithms with the same colors as in the line plots
    handles = [plt.Line2D([0], [0], color=sns.color_palette("colorblind", n_colors=len(mapping["algorithm"]))[i], lw=4) for i in range(len(mapping["algorithm"]))][::-1]
    labels = [mapping["algorithm"][algo] for algo in mapping["algorithm"]][::-1]
    # set up teh hanldes styles as dashed for the optimal algorithm and solid for the rest
    for i in range(len(handles)):
        if i==0:
            handles[i].set_linestyle('--')
        else:
            handles[i].set_linestyle('-')
    # do not include axes in the figure and save only the legend
    plt.gca().set_axis_off()
    plt.legend(handles=handles, labels=labels, title='', ncol=2, loc='center', fontsize=80, frameon=True)
    plt.savefig(f"figures/legend/regret_legend.pdf", bbox_inches='tight')
    plt.clf()        

if __name__ == "__main__":
    configs = [
        ("configs/human_alignment.json", "human_alignment"),
        ("configs/human_ai_interactions.json", "human_ai_interactions"),
    ]
    for config_file, fig_path_id in tqdm(configs):
        regret_df_full, config = compute_regret_df(config_file)
        ds_names = regret_df_full['dataset'].unique()
        add_legend = False 
        for ds_name in tqdm(ds_names):
            
            fig_path = f"figures/{fig_path_id}/{config['num_iterations']}/{ds_name}"
            p_y_given_b_h_b(f"{config['data_dir']}/{ds_name}.csv", config, fig_path)    
        
            regret_df = regret_df_full[regret_df_full['dataset'] == ds_name].copy()
            avg_regret_path = f"{config['output_dir']}/avg_regret/{config['num_iterations']}/{ds_name}"
            fig_path = f"figures/{fig_path_id}/{config['num_iterations']}/{ds_name}"
            regret(regret_df, avg_regret_path, fig_path, add_legend=add_legend, config=config, ds_name=ds_name)
        
    
    plot_legend_path = f"figures/legend"
    os.makedirs(plot_legend_path, exist_ok=True)
    
    regret_legend()
    alignment_legend()

