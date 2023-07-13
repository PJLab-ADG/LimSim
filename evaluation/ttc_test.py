import matplotlib.pyplot as plt

plt.style.use("seaborn-poster")
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from collision_statistics import compute_time_to_collision


def plot_ttc_version_1(ttc_statistics) -> None:
    plt.figure(figsize=(40, 4))
    # fig, ax = plt.subplots()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    font1 = {'family': 'fantasy', 'size': 20}

    x, y = ttc_statistics[:, 0], ttc_statistics[:, 1]
    # ttc_statistics[:, 1] = np.minimum(ttc_statistics[:, 1], 10)
    plt.plot(x, y, linewidth=2, color="#70a1ff")
    plt.xlabel("Timestamp", fontdict=font1)
    plt.ylabel("TTC (s)", fontdict=font1)
    plt.ylim([0, 10.1])
    x_ticks = [2500, 2700, 2800, 3000, 3500, 4000, 4500, 5000]
    plt.xticks(x_ticks, fontsize=16)
    y_ticks = [0, 2, 4, 6, 8, 10]
    y_labels = ['0', '2', '4', '6', '8', '10+']
    plt.yticks(y_ticks, y_labels, fontsize=16)
    # plt.grid("on")
    plt.show()


def plot_ttc_version_2(ttc_statistics) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x, y = ttc_statistics[:, 0], ttc_statistics[:, 1]
    ax.plot(x, y, linewidth=2, label="ttc")

    ax.legend()

    # partial enlargement
    axins = inset_axes(ax,
                       width="40%",
                       height="30%",
                       loc='lower right',
                       bbox_to_anchor=(0.1, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)

    axins.plot(
        x,
        y,
        linewidth=2,
    )

    zone_left = 256
    zone_right = 260

    x_ratio = 0.5  # x轴显示范围的扩展比例
    y_ratio = 0.5  # y轴显示范围的扩展比例

    xlim0 = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim1 = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    ylim0 = 2.5
    ylim1 = 20 * y_ratio

    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # connect the zoom-in area
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    plt.show()


def plot_ttc_version_3(ttc_statistics):
    x, y = ttc_statistics[:, 0], ttc_statistics[:, 1]
    y = np.minimum(y, 10)

    interval_1 = [0, 254]
    interval_2 = [interval_1[1], 262]
    interval_3 = [interval_2[1], 2209]
    interval_4 = [interval_3[1], 2250]
    interval_5 = [interval_4[1], x.size]

    enlarge_ratio = 3
    kwargs = {'width_ratios': [1, enlarge_ratio, 1, enlarge_ratio, 1]}

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,
                                                  5,
                                                  figsize=(40, 6),
                                                  gridspec_kw=kwargs)

    # fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    ax1.plot(x[interval_1[0]:interval_1[1]], y[interval_1[0]:interval_1[1]],
             '-*')
    ax2.plot(x[interval_2[0]:interval_2[1]], y[interval_2[0]:interval_2[1]],
             '-*')
    ax3.plot(x[interval_3[0]:interval_3[1]], y[interval_3[0]:interval_3[1]],
             '-*')
    ax4.plot(x[interval_4[0]:interval_4[1]], y[interval_4[0]:interval_4[1]],
             '-*')
    ax5.plot(x[interval_5[0]:interval_5[1]], y[interval_5[0]:interval_5[1]],
             '-*')

    # set the spines style between axes
    ax1.yaxis.tick_left()
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([0, 10.1])
    ax1.set_xticks([])

    ax2.spines['left'].set_linestyle((0, (10, 10)))
    ax2.spines['right'].set_linestyle((0, (10, 10)))
    ax2.set_yticks([])
    ax2.set_ylim([0, 10.1])

    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_yticks([])
    ax3.set_ylim([0, 10.1])
    ax3.set_xticks([])

    # ax4.spines['left'].set_visible(False)
    ax4.spines['left'].set_linestyle((0, (10, 10)))
    ax4.spines['right'].set_linestyle((0, (10, 10)))
    ax4.set_yticks([])
    ax4.set_ylim([0, 10.1])

    ax5.spines['left'].set_visible(False)
    # ax5.spines['right'].set_visible(False)
    ax5.yaxis.tick_right()
    ax5.set_ylim([0, 10.1])
    ax5.set_xticks([])

    # Make the spacing between the two axes a bit smaller
    plt.subplots_adjust(wspace=0.05)
    # plt.ylim([0, 10])

    d = 0.02  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    left_x, left_y = np.array([-d, d]), np.array([-d, d])
    right_x, right_y = np.array([1 - d, 1 + d]), np.array([1 - d, 1 + d])
    shifts = (1 - 1.0 / enlarge_ratio) * d * np.array([1, -1])

    ax1.plot(right_x, left_y, **kwargs)  # top-left diagonal
    ax1.plot(right_x, right_y, **kwargs)  # bottom-left diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot(left_x + shifts, left_y, **kwargs)
    ax2.plot(left_x + shifts, right_y, **kwargs)
    ax2.plot(right_x + shifts, left_y, **kwargs)  # top-right diagonal
    ax2.plot(right_x + shifts, right_y, **kwargs)  # bottom-right diagonal

    kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
    ax3.plot(left_x, left_y, **kwargs)  # top-right diagonal
    ax3.plot(left_x, right_y, **kwargs)  # bottom-right diagonal
    ax3.plot(right_x, left_y, **kwargs)  # top-right diagonal
    ax3.plot(right_x, right_y, **kwargs)  # bottom-right diagonal

    kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
    ax4.plot(left_x / enlarge_ratio, left_y, **kwargs)  # top-right diagonal
    ax4.plot(left_x / enlarge_ratio, right_y,
             **kwargs)  # bottom-right diagonal
    ax4.plot(right_x + shifts, left_y, **kwargs)  # top-right diagonal
    ax4.plot(right_x + shifts, right_y, **kwargs)  # bottom-right diagonal

    kwargs.update(transform=ax5.transAxes)  # switch to the bottom axes
    ax5.plot(left_x, left_y, **kwargs)  # top-right diagonal
    ax5.plot(left_x, right_y, **kwargs)  # bottom-right diagonal

    plt.show()


def plot_ttc_version_4(ttc_statistics):
    x, y = ttc_statistics[:, 0], ttc_statistics[:, 1]
    fake_inf = 11.5
    y = np.minimum(y, fake_inf)

    interval_1 = [0, 252]
    interval_2 = [interval_1[1], 264]
    interval_3 = [interval_2[1], 2209]
    interval_4 = [interval_3[1], 2250]
    interval_5 = [interval_4[1], x.size]
    intervals = [interval_1, interval_2, interval_3, interval_4, interval_5]

    # remove data
    y[interval_3[0]:interval_3[1]] = fake_inf

    # width of each subplot
    enlarge_ratios = [1, 8, 1, 8, 1]
    kwargs = {'width_ratios': enlarge_ratios}

    fig, axes = plt.subplots(1, 5, figsize=(40, 6), gridspec_kw=kwargs)


    d = 0.02  # proportion of vertical to horizontal extent of the slanted line
    left_x, left_y = np.array([-d, d]), np.array([-d, d])
    right_x, right_y = np.array([1 - d, 1 + d]), np.array([1 - d, 1 + d])
    shift_y = -0.00 * np.ones(2)

    for index, (ax, interval) in enumerate(zip(axes, intervals)):
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        shift = (1.0 - 1.0 / enlarge_ratios[index]) * d * np.array([1, -1])
        if index == 1 or index == 3:
            ax.plot(x[interval[0]:interval[1]], y[interval[0]:interval[1]], '-*')
        else:
            ax.plot(x[interval[0]:interval[1]], y[interval[0]:interval[1]])

        # set the spines style between axes
        if index == 0:
            ax.yaxis.tick_left()
            ax.spines['right'].set_visible(False)
            ax.set_ylim([0, 12])
            ax.set_xticks([])
        elif index == 4:
            ax.spines['left'].set_visible(False)
            ax.yaxis.tick_right()
            ax.set_ylim([0, 12])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            if index == 1 or index == 3:
                ax.spines['left'].set_linestyle((0, (10, 10)))
                ax.spines['right'].set_linestyle((0, (10, 10)))
                # ax.spines['left'].set_visible(False)
                # ax.spines['right'].set_visible(False)
            else:
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([0, 12])
        # ax.spines['top'].set_visible(False)

        # plot cut-out slanted lines
        if index == 0:
            ax.plot(right_x + shift, left_y, **kwargs)  # top-left diagonal
            ax.plot(right_x + shift, right_y + shift_y,
                    **kwargs)  # bottom-left diagonal
        elif index == 4:
            ax.plot(left_x + shift, left_y, **kwargs)  # top-right diagonal
            ax.plot(left_x + shift, right_y + shift_y,
                    **kwargs)  # bottom-right diagonal
        else:
            ax.plot(left_x + shift, left_y, **kwargs)  # top-right diagonal
            ax.plot(left_x + shift, right_y + shift_y,
                    **kwargs)  # bottom-right diagonal
            ax.plot(right_x + shift, left_y, **kwargs)  # top-right diagonal
            ax.plot(right_x + shift, right_y + shift_y,
                    **kwargs)  # bottom-right diagonal

    # axis label
    ticks = [0, 2, 4, 6, 8, 10, 12]
    axes[0].set_yticks(ticks)
    labels = ['0', '2', '4', '6', '8', '10+', '']
    axes[0].set_yticklabels(labels)

    axes[0].set_ylabel("TTC (s)", fontsize=16)
    axes[2].set_xlabel("Frame", fontsize=16)
    axes[2].xaxis.set_label_coords(0.5, -0.1)

    plt.subplots_adjust(wspace=0.02)
    plt.show()


def main():
    ttc_statistics = compute_time_to_collision("../egoTrackingTest.db")
    plot_ttc_version_4(ttc_statistics)


if __name__ == '__main__':
    main()