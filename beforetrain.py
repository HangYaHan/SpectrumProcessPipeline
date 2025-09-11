
from pre_funcs import *
import pandas as pd

if __name__ == "__main__":
    pic_path = "./data/pics/"
    spec_path = "./data/specs/"
    regioncounts = 18

    spec_remove_lines_path = "./result/RemovedLines/"
    recaculate_to_int_path = "./result/RecaculatedToInt/"
    cut_to_visible_range_path = "./result/CutToVisibleRange/"

    # --- PIPELINE ---
    # rename_and_compare(pic_path, spec_path)
    # remove_lines(spec_path, spec_remove_lines_path, num_lines=14)
    # recalculate_to_int(spec_remove_lines_path, recaculate_to_int_path)
    # cut_to_visible_range(recaculate_to_int_path, cut_to_visible_range_path, min_range=400, max_range=800)
    # select_regions(pic_path+"/0009.jpg", "rois.txt", regioncounts)

    # --- GREY CALC ---
    # calc_rois_gray_to_csv("rois.txt", pic_path, "grey.csv")
    # df = pd.read_csv("grey.csv")
    # print(df.shape)

    # --- SPECTRUM CALC ---
    # spectrum_to_csv(cut_to_visible_range_path, "spectrum.csv")
    # df = pd.read_csv("spectrum.csv")
    # print(df.shape)

