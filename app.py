#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/9 23:45
# @Author  : Bo and Ken

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import joblib
import xlrd
import pandas as pd
import streamlit as st
import os
import shutil         

# calibration range inpolygon detection
def inpoly_detector(data_use: np.ndarray,
                    book: str = 'allexps_all.xlsx',
                    sheet_index: int = 2) -> np.ndarray:
    """
    Returns an int array (1 = inside polygon for all projections, 0 = outside).

    Parameters
    ----------
    data_use : np.ndarray
        Your sample matrix (rows × ≥9 cols).
    book : str
        Path to allexps_all.xlsx (default assumes it sits beside app.py).
    sheet_index : int
        Zero‑based index of the worksheet that holds the melt data.
        The third sheet ⇒ 2.
    """
    import pandas as pd
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull

    # --- read experimental “melt” data from the 3rd worksheet ---------------
    data_exp = pd.read_excel(book,
                             sheet_name=sheet_index,   # ← 3rd sheet
                             engine='openpyxl').to_numpy()

    # --- polygon‑inclusion test (vectorised) --------------------------------
    proj_cols = [1, 2, 3, 5, 6, 7, 8]      # MATLAB [2 3 4 6 7 8 9] → 0‑based
    inside = np.ones(len(data_use), dtype=bool)

    for i in proj_cols:
        pts_exp = np.column_stack((data_exp[:, 0], data_exp[:, i]))
        hull    = ConvexHull(pts_exp)
        poly    = Path(pts_exp[hull.vertices])
        inside &= poly.contains_points(
                      np.column_stack((data_use[:, 0], data_use[:, i])))

    return inside.astype(int)


# run plag-sat classifier
def run_plgsat_classifier(X: np.ndarray) -> np.ndarray:
    rf_use_plgsat = joblib.load(r'rfmodels/classifier_plgsat')
    return rf_use_plgsat.predict(X[:, :10])               # returns 0 / 1

# import data as numpy matrix from excel file
def import_excel_matrix(path, i):
    try:
        table = xlrd.open_workbook(path).sheets()[i]
        row = table.nrows - 2
        # Determine the number of columns by checking the first non-header row
        first_data_row = table.row_values(2)
        col = len([cell for cell in first_data_row if cell != ''])
        data_matrix = np.zeros((row, col))
        for i in range(col):
            cols_raw = table.col_values(i)
            cols = np.matrix(cols_raw[2:])
            data_matrix[:, i] = cols
        return data_matrix
    except:
        st.error('Please check your upload file!')



# write data into excel file
def save_excel(data, col, path=''):
    dataFrame = pd.DataFrame(data)
    with pd.ExcelWriter(path, mode='a', if_sheet_exists='overlay') as writer:
        dataFrame.to_excel(writer, sheet_name='Sheet1', float_format='%.6f', index=False, startrow=2, startcol=col,
                           header=None)


def main(input_file, model_file,id):
    # path for input file
    # input_file = r'Template_input.xlsx'

    # import data from input excel

    data = import_excel_matrix(input_file, 0)
    # degin input data propcessing
    X = data

    # renormalize data to 100 wt.%
    X_wid = X.shape[1]
    if X_wid == 10:
        X = X / np.sum(X, axis=1, keepdims=True) * 100
    elif X_wid == 20:
        X[:, :10] = X[:, :10] / np.sum(X[:, :10], axis=1, keepdims=True) * 100
        X[:, 10:] = X[:, 10:] / np.sum(X[:, 10:], axis=1, keepdims=True) * 100

    # choose the random forest model
    # model_file = r'rfmodels/hygro_liq-afterplgin'
    rf_use = joblib.load(model_file)

    # run the model to give predicted values
    y_predict = rf_use.predict(X)
    y_predict = y_predict.reshape((len(y_predict), 1))

    # the importance ranking for future use
    # rf_use.feature_importances_

    # path for output file
    output_file = 'downloads/'+id+'_output.xlsx'

    # detect with type of model is used
    output_col = 0
    if 'hygro' in model_file:
        output_col = 20
    elif 'thermo' in model_file:
        output_col = 21
    elif 'baro' in model_file:
        output_col = 22

    # write the predicted values and original data to output excel

    save_excel(y_predict, output_col, output_file)
    save_excel(data, 0, output_file)

    # get all the model names
    data_name = ['liq', 'liq-outolonly', 'liq-afterplgin', 'ol', 'cpx', 'plg', 'opx', 'amph', 'ilm', 'mag', 'sp', 'grt']
    data_name_raw = ['liq', 'ol', 'cpx', 'plg', 'opx', 'amph', 'ilm', 'mag', 'sp', 'grt']
    for i in range(10):
        for j in range(10):
            if j > i:
                data_name.append(data_name_raw[i] + '-' + data_name_raw[j])

    # find the index of model in calibration data
    parts = model_file.split('_', -1)
    model_name = parts[-1]
    model_index = data_name.index(model_name)

    # import experimental calibration data
    input_file_exp = 'allexps_all.xlsx'
    data_exp = import_excel_matrix(input_file_exp, model_index)

    # set up plot parameters
    label_cb = ''
    if 'hygro' in model_file:
        label_cb = 'H$_2$O (wt.%)'
    elif 'thermo' in model_file:
        label_cb = 'T (°C)'
    elif 'baro' in model_file:
        label_cb = 'P (kbar)'
    c_map = 'hsv'
    alpha = 0.3

    # plot Fig.1: histogram of predicted values
    fig1 = plt.figure(figsize=(10,2))
    plt.hist(y_predict, color='wheat', edgecolor='black', linewidth=0.9)
    plt.xlabel(label_cb, fontweight="bold")
    plt.ylabel('Frequency', fontweight="bold")

    # st.pyplot(fig1)

    # plot Fig.2: Haker diagram
    fig2 = plt.figure(figsize=(10,7.5))

    plt.subplot(3, 3, 1)
    plt.scatter(data_exp[:, 0], data_exp[:, 1], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 1], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    cbar = plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('TiO$_2$ (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 2)
    plt.scatter(data_exp[:, 0], data_exp[:, 2], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 2], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('Al$_2$O$_3$ (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 3)
    plt.scatter(data_exp[:, 0], data_exp[:, 3], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 3], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('FeOT (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 4)
    plt.scatter(data_exp[:, 0], data_exp[:, 4], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 4], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('MnO (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 5)
    plt.scatter(data_exp[:, 0], data_exp[:, 5], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 5], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('MgO (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 6)
    plt.scatter(data_exp[:, 0], data_exp[:, 6], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 6], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('CaO (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 7)
    plt.scatter(data_exp[:, 0], data_exp[:, 7], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 7], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('Na$_2$O (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 8)
    plt.scatter(data_exp[:, 0], data_exp[:, 8], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 8], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('K$_2$O (wt.%)', fontweight="bold")

    plt.subplot(3, 3, 9)
    plt.scatter(data_exp[:, 0], data_exp[:, 9], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
    scatter = plt.scatter(data[:, 0], data[:, 9], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                          label='Input Data')
    plt.colorbar(scatter, label=label_cb)
    plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
    plt.ylabel('P$_2$O$_5$ (wt.%)', fontweight="bold")

    plt.tight_layout()
    # plt.show()
    # st.pyplot(fig2)
    # if model contains two components, plot Fig.3: Haker diagram
    fig3 = None
    if X_wid == 20:
        fig3 = plt.figure(figsize=(10,7.5))

        plt.subplot(3, 3, 1)
        plt.scatter(data_exp[:, 10], data_exp[:, 11], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 0], data[:, 11], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        cbar = plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('TiO$_2$ (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 2)
        plt.scatter(data_exp[:, 10], data_exp[:, 12], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 0], data[:, 12], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('Al$_2$O$_3$ (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 3)
        plt.scatter(data_exp[:, 10], data_exp[:, 13], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 13], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('FeOT (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 4)
        plt.scatter(data_exp[:, 10], data_exp[:, 14], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 14], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('MnO (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 5)
        plt.scatter(data_exp[:, 10], data_exp[:, 15], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 15], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('MgO (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 6)
        plt.scatter(data_exp[:, 10], data_exp[:, 16], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 16], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('CaO (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 7)
        plt.scatter(data_exp[:, 10], data_exp[:, 17], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 17], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('Na$_2$O (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 8)
        plt.scatter(data_exp[:, 10], data_exp[:, 18], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 18], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('K$_2$O (wt.%)', fontweight="bold")

        plt.subplot(3, 3, 9)
        plt.scatter(data_exp[:, 10], data_exp[:, 19], linewidths=0, color='grey', alpha=alpha, label='Calibration Data')
        scatter = plt.scatter(data[:, 10], data[:, 19], edgecolors='black', linewidths=0.2, c=y_predict, cmap=c_map,
                              label='Input Data')
        plt.colorbar(scatter, label=label_cb)
        plt.xlabel('SiO$_2$ (wt.%)', fontweight="bold")
        plt.ylabel('P$_2$O$_5$ (wt.%)', fontweight="bold")

        plt.tight_layout()
        # plt.show()
        # st.pyplot(fig3)
    return fig1, fig2, fig3


import os, shutil  


def copy_files(path,id):  
    for foldName, subfolders, filenames in os.walk(path):  
        for filename in filenames:  
            if filename.endswith('Template_output.xlsx'):  
                new_name = filename.replace('Template_output.xlsx', id+'_output.xlsx')  
                shutil.copyfile(os.path.join(foldName, filename), os.path.join(foldName, new_name))  
                print(filename, "copied as", new_name)



st.title('Hygro-Thermobarometer by Bo and Jagoutz 2024')

tab_calc, tab_plgsat, tab_info = st.tabs(['**Calc**', '**Plg‑sat.**', '**Info**'])  # :contentReference[oaicite:0]{index=0}

with tab_calc:
    st.write(
        '***Ranking of all possible hygro-thermobarometer pairs. RMSE are averaged from 100 trials of calibration of 80% train data and 20% test data.***')
    meter_rank = pd.read_excel('meter_rank.xlsx', sheet_name='Sheet1')
    # print(meter_rank)
    st.dataframe(meter_rank, hide_index=True)

    st.subheader('***Step1: What do you want to predict?***')

    question1 = st.selectbox('', ['Water (wt.%)', 'Temperature (°C)', 'Pressure (kbar)'])
    if question1 == 'Water (wt.%)':
        ex_model = 'hygro'
    elif question1 == 'Pressure (kbar)':
        ex_model = 'baro'
    else:
        ex_model = 'thermo'

    st.subheader('***Step 2: Which liquid/mineral pair do you want to use?***')

    question2 = st.selectbox('', list(meter_rank['Pairs']))
    model_file = "rfmodels/" + ex_model + "_" + question2


    # @st.cache_data
    # def convert_df(df):
    #     return df.to_csv().encode("utf-8")


    # Template_input = pd.read_excel('Template_input.xlsx')
    # csv = convert_df(Template_input)

    # st.subheader('***Step 3: Please download the template***')
    # st.download_button(
    #     label="**Template_input.csv**",
    #     data=csv,
    #     file_name="Template_input.csv",
    #     mime="text/csv",
    # )

    st.subheader('***Step 3: Please download the template***')
    with open('Template_input.xlsx', 'rb') as my_file:
        st.download_button(label='**Template_input.xlsx**', data=my_file, file_name='Template_input.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.subheader('***Step 4: Please upload your data using the template***')
    upload_file = st.file_uploader('')

    if upload_file:
        import uuid
        id = str(uuid.uuid1())
        with open(os.path.join("uploads", id+'_'+upload_file.name), "wb") as f:
            f.write(upload_file.getbuffer())

        with st.spinner('***Calculation…***'):
            if upload_file:
                copy_files('downloads',id)
                f1, f2, f3 = main(os.path.join("uploads", id+'_'+upload_file.name), model_file,id)
                st.success('***Calculation is complete***')
                with open("downloads/"+id+"_output.xlsx", "rb") as file:
                    st.subheader('***Step 5: Please download your results***')
                    st.download_button(label='**Template_output.xlsx**', data=file, file_name='Template_output.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                st.write(
                    '***The results predicted by data which falls out of the compositional range of experimental data are not reliable. The experimental data used for calibration is shown as grey circle, and  your data is shown as circle with color-coded.***')
                st.write('**Figure 1: Distribution of predicted values**')
                st.pyplot(f1)
                st.write('**Figure 2: Major elemental covariations of Component 1**')
                st.pyplot(f2)
                if f3:
                    st.write('**Figure 3: Major elemental covariations of  Component 2**')
                    st.pyplot(f3)

# ─────────────────────────────  “Plg‑sat.” TAB  ─────────────────────────────
with tab_plgsat:

    st.write('***Plagioclase‑saturation checker***')

    # ---------- Step1 – download template ------------------------------------
    st.subheader('***Step1: Please download the template***')
    with open('Template_input_plgsat.xlsx', 'rb') as tpl:      # note new file
        st.download_button(label='**Template_input_plgsat.xlsx**',
                           data=tpl,
                           file_name='Template_input_plgsat.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # ---------- Step2 – upload data ------------------------------------------
    st.subheader('***Step2: Please upload your data using the template***')
    upl_file = st.file_uploader('', type=['xlsx'])

# In the plg-sat tab section, replace the current file handling code with this:

if upl_file:
    st.subheader('***Step3: Testing whether the uploaded liquid composition is (1) Plag saturated and (2) within the calibration range of our experimental data***')
  
    import uuid, os
    sess_id = str(uuid.uuid4())
    up_path = os.path.join('uploads', f'{sess_id}_{upl_file.name}')
    with open(up_path, 'wb') as f:
        f.write(upl_file.getbuffer())

    with st.spinner('***Running classifier & polygon test…***'):            
        # 1) Copy the template output file to preserve formatting
        template_output_path = 'downloads/Template_output_plgsat.xlsx'
        dl_name = f'{sess_id}_plgsat_output.xlsx'
        dl_path = os.path.join('downloads', dl_name)
        
        # Make sure the template output file exists, then copy it
        if os.path.exists(template_output_path):
            shutil.copyfile(template_output_path, dl_path)
        else:
            st.error("Template output file not found. Using uploaded file as base.")
            shutil.copyfile(up_path, dl_path)
        
        # 2) read the numeric matrix from the uploaded file
        X = import_excel_matrix(up_path, 0)
        
        # 3) renormalise to 100 wt.%
        X_wid = X.shape[1]
        if X_wid == 10:
            X = X / np.sum(X, axis=1, keepdims=True) * 100
        elif X_wid == 20:
            X[:, :10]  = X[:, :10]  / np.sum(X[:, :10],  axis=1, keepdims=True) * 100
            X[:, 10:]  = X[:, 10:]  / np.sum(X[:, 10:],  axis=1, keepdims=True) * 100
        
        # 4) Save the input data first to the template copy
        save_excel(X, 0, dl_path)
        
        # 5) run the two classifiers
        out_rf   = run_plgsat_classifier(X).reshape(-1, 1)
        out_poly = inpoly_detector(X).reshape(-1, 1)
        
        # 6) append the results to the workbook
        save_excel(out_rf,   X_wid,     dl_path)
        save_excel(out_poly, X_wid + 1, dl_path)


        st.success('***Calculation is complete***')

        # ---------- Step3 – download results ---------------------------------
        st.subheader('***Step4: Please download your results***')
        with open(dl_path, 'rb') as fh:
            st.download_button(label='**Template_output_plgsat.xlsx**',
                               data=fh,
                               file_name='Template_output_plgsat.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # quick stats (optional, matches Calc tab style)
        st.write(f'Rows flagged by the RF model: **{out_rf.sum()}**')
        st.write(f'Rows inside every convex‑hull projection: **{out_poly.sum()}**')


with tab_info:
    col1, col2 = st.columns([6, 4])
    with col1:
        st.subheader('***Effect of Water on Magmatic Evolution Systematics, 2024***')
    with col2:
        st.image('dog.png')
