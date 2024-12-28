import os.path as osp
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval


def dump_to_json_file(obj, dst_file):
    assert dst_file.endswith('.json')
    with open(dst_file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def df_to_xlsx(df, dst_file, *, index=False, text_wrap=False):

    assert isinstance(df, pd.DataFrame)
    assert dst_file.endswith('.xlsx')

    # ------ output to xlsx
    writer = pd.ExcelWriter(dst_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=index)
    workbook = writer.book

    # ------ freeze the first row
    worksheet = writer.sheets['Sheet1']
    worksheet.freeze_panes(df.columns.nlevels, 0)

    # ------ get the number of rows and columns
    if index:
        num_cols = df.shape[1] + df.index.nlevels
    else:
        num_cols = df.shape[1]
    num_rows = df.shape[0] + df.columns.nlevels

    # ------ set all cell boundary line to black
    cell_format = workbook.add_format({'bottom': True, 'right': True})
    worksheet.conditional_format(0, 0, num_rows, num_cols, {
        'type': 'no_blanks',
        'format': cell_format
    })
    # ------ set all cells to wrap text
    if text_wrap:
        cell_format = workbook.add_format({'text_wrap': True})
        worksheet.set_column(0, num_cols, None, cell_format)
    writer.close()


def df_to_list_of_dict(df, all_to_str=False, keys_to_str=False):
    assert isinstance(df, pd.DataFrame)

    if all_to_str:
        df = df.astype(str)

    ret_list = df.to_dict(orient='records')

    if keys_to_str:
        new_list = []
        for cur_dict in ret_list:
            new_dict = {str(k): v for k, v in cur_dict.items()}
            new_list.append(new_dict)
        ret_list = new_list

    return ret_list


def literal_eval_with_None(src):
    if src is None or pd.isna(src):
        return None
    return literal_eval(src)


def list_of_list_to_list_of_str(src):
    assert isinstance(src, list)
    return [str(_) for _ in src]


def list_of_str_to_list_of_list(src):
    assert isinstance(src, list)
    if not isinstance(src[0], list):
        return src

    return [literal_eval_with_None(_) for _ in src]


def draw_hist_mat(
        x,
        y,
        *,
        x_label,
        y_label,
        dst_dir,
        dst_prefix,
        x_bins=7,
        y_bins=5,
        figsize=(6, 5),
        hist_names=['num', 'prob'],
        range=None,
        title=None,
):
    x = np.array(x)
    y = np.array(y)

    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=[x_bins, y_bins],
        range=range,
    )
    hist_n = hist.T  # For proper visualization, transpose the result

    # normlize the hist
    hist_p = hist_n / hist_n.sum()

    hist_dict = {
        'num': hist_n,
        'prob': hist_p,
    }
    fmt_dict = {
        'num': '.0f',
        'prob': '.2f',
    }
    if title is None:
        title = f'{osp.basename(dst_dir)}__{dst_prefix}'
    xticklabels = [f'{x:.2f}' for x in x_edges]
    yticklabels = [f'{y:.2f}' for y in y_edges]
    for hist_name in hist_names:
        hist = hist_dict[hist_name]
        plt.figure(figsize=figsize)
        sns.heatmap(hist,
                    annot=True,
                    fmt=fmt_dict[hist_name],
                    cmap='coolwarm',
                    center=0)

        plt.xticks(np.arange(x_bins + 1), xticklabels)
        plt.yticks(np.arange(y_bins + 1), yticklabels)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(osp.join(dst_dir, f'{dst_prefix}_{hist_name}.png'))

    return [
        osp.join(dst_dir, f'{dst_prefix}_{hist_name}.png')
        for hist_name in hist_names
    ]


def compare_chessboard(
        *,
        dst_dir,
        src_df,
        dst_df,
        compare_info,
        xlabel,
        ylabel,
        xrange,
        yrange,
        bins_list,
        figsize=(6, 6),
        title=None,
):

    dump_to_json_file(compare_info, osp.join(dst_dir, 'compare_info.json'))

    assert set(src_df.columns) == set(['x', 'y', 'qid'])
    assert set(dst_df.columns) == set(['x', 'y', 'qid'])

    src_df.rename(columns={'x': 'src_x', 'y': 'src_y'}, inplace=True)
    dst_df.rename(columns={'x': 'dst_x', 'y': 'dst_y'}, inplace=True)

    df = pd.merge(src_df, dst_df, on='qid', how='inner')

    df['dx'] = df.apply(lambda x: x['dst_x'] - x['src_x'], axis=1)
    df['dy'] = df.apply(lambda x: x['dst_y'] - x['src_y'], axis=1)

    df.to_csv(osp.join(dst_dir, 'dx_dy.csv'), index=False)
    dump_to_json_file(df_to_list_of_dict(df), osp.join(dst_dir, 'dx_dy.json'))

    draw_hist_mat(
        df['dx'],
        df['dy'],
        x_label='dx',
        y_label='dy',
        dst_dir=dst_dir,
        dst_prefix='dx_dy',
        x_bins=10,
        y_bins=8,
        hist_names=['num', 'prob'],
        range=[(-0.5, 0.5), (-0.5, 0.5)],
        title=title,
    )

    for xbins, ybins in bins_list:

        dst_prefix = osp.join(dst_dir, f'grid_{xbins}x{ybins}')

        xedges = np.linspace(xrange[0], xrange[1], xbins + 1)
        yedges = np.linspace(yrange[0], yrange[1], ybins + 1)
        # add eps to last element to include the rightmost edge
        xedges[-1] = xrange[1] + 1e-8
        yedges[-1] = yrange[1] + 1e-8

        chessboard_info = {
            'compare_info': compare_info,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xrange': xrange,
            'yrange': yrange,
            'xbins': xbins,
            'ybins': ybins,
            'xedges': xedges.tolist(),
            'yedges': yedges.tolist(),
            'title': title,
        }

        assert isinstance(compare_info, dict)
        dataset = compare_info['dataset']
        src_model = compare_info['src_model']
        dst_model = compare_info['dst_model']
        src_prompting = compare_info['src_prompting']
        dst_prompting = compare_info['dst_prompting']
        src_lang = compare_info['src_lang']
        dst_lang = compare_info['dst_lang']

        grids_info = {'chessboard': chessboard_info}
        grids_dfs = {}
        for i in range(xbins):
            for j in range(ybins):
                xlo, xhi = xedges[i], xedges[i + 1]
                ylo, yhi = yedges[j], yedges[j + 1]

                grid_df = df[(df['src_x'] >= xlo) & (df['src_x'] < xhi) &
                             (df['src_y'] >= ylo) & (df['src_y'] < yhi)]
                grids_dfs[f'grid_{i}_{j}'] = grid_df

                grid_info = {
                    'i': i,
                    'j': j,
                    'xrange': (xlo, xhi),
                    'yrange': (ylo, yhi),
                    'src_num': len(grid_df),
                }
                if grid_df.empty:
                    grids_info[f'grid_{i}_{j}'] = grid_info
                    continue

                dx_mean = grid_df['dx'].mean()
                dy_mean = grid_df['dy'].mean()
                dx_std = grid_df['dx'].std()
                dy_std = grid_df['dy'].std()

                grid_info.update({
                    'dx_mean': dx_mean,
                    'dy_mean': dy_mean,
                    'dx_std': dx_std,
                    'dy_std': dy_std,
                })
                grids_info[f'grid_{i}_{j}'] = grid_info

        assert df.shape[0] == sum([
            grids_info[f'grid_{i}_{j}']['src_num'] for i in range(xbins)
            for j in range(ybins)
        ])

        dump_to_json_file(grids_info, dst_prefix + '_info.json')

        # -------------------------------- draw (dx_mean, dy_mean) vectors
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(top=0.85)
        fig.text(0.05,
                 0.95,
                 f'dataset: {dataset}',
                 ha='left',
                 va='center',
                 fontsize=10)
        fig.text(0.05,
                 0.90,
                 f'src:{src_model}({src_prompting})({src_lang})',
                 ha='left',
                 va='center',
                 fontsize=10)
        fig.text(0.05,
                 0.85,
                 f'dst:{dst_model}({dst_prompting})({dst_lang})',
                 ha='left',
                 va='center',
                 fontsize=10)

        # draw grids
        for i in range(xbins):
            ax.axvline(xedges[i], color='black', linewidth=0.5)
        for j in range(ybins):
            ax.axhline(yedges[j], color='black', linewidth=0.5)

        for i in range(xbins):
            for j in range(ybins):
                grid_info = grids_info[f'grid_{i}_{j}']
                if grid_info['src_num'] == 0:
                    continue
                x = (xedges[i] + xedges[i + 1]) / 2
                y = (yedges[j] + yedges[j + 1]) / 2
                dx_mean = grid_info['dx_mean']
                dy_mean = grid_info['dy_mean']
                ax.arrow(
                    x,
                    y,
                    dx_mean,
                    dy_mean,
                    head_width=0.02,
                    head_length=0.02,
                    fc='r',
                    ec='r',
                )

                ax.text(
                    x,
                    y,
                    f'{grid_info["src_num"]}',
                    color='black',
                    ha='center',
                    va='center',
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.invert_yaxis()

        plt.title(title)
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.savefig(dst_prefix + '_dx_dy_mean.png')
        plt.close()

        # -------------------------------- draw (dx, dy) for each grid
        fig, axes = plt.subplots(ybins, xbins, figsize=(12, 12))
        if xbins == 1 and ybins == 1:
            axes = np.array([[axes]])
        elif xbins == 1:
            axes = np.array([[ax] for ax in axes])
        elif ybins == 1:
            axes = np.array([axes])

        fig.subplots_adjust(top=0.85)
        fig.text(0.1,
                 0.96,
                 f'src:{src_model}({src_prompting})({src_lang})',
                 ha='left',
                 va='center',
                 fontsize=10)
        fig.text(0.1,
                 0.92,
                 f'dst:{dst_model}({dst_prompting})({dst_lang})',
                 ha='left',
                 va='center',
                 fontsize=10)
        fig.text(0.1,
                 0.88,
                 f'dataset:{dataset}; xlabel:{xlabel}; ylabel:{ylabel}',
                 ha='left',
                 va='center',
                 fontsize=10)

        for i in range(xbins):
            for j in range(ybins):
                ax = axes[j, i]
                grid_info = grids_info[f'grid_{i}_{j}']
                if grid_info['src_num'] == 0:
                    ax.axis('off')
                    continue

                grid_df = grids_dfs[f'grid_{i}_{j}']
                ax.scatter(grid_df['dx'], grid_df['dy'], s=1)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.invert_yaxis()
                ax.grid(True)
                ax.set_aspect('equal')
                ax.set_title(f'src_num = {grid_info["src_num"]}')

        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.savefig(dst_prefix + '_dx_dy.png')
        plt.close()
