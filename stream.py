import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from bokeh.plotting import figure, show
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap
# filter
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import streamlit.components.v1 as components

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# MPR
from mp_api.client import MPRester
from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    #https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters")

    #if not modify:
    #    return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def periodic_table():

        periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
        groups = [str(x) for x in range(1, 19)]

        df = elements.copy()
        df["atomic mass"] = df["atomic mass"].astype(str)
        df["group"] = df["group"].astype(str)
        df["period"] = [periods[x-1] for x in df.period]
        df = df[df.group != "-"]
        df = df[df.symbol != "Lr"]
        df = df[df.symbol != "Lu"]

        cmap = {
            "alkali metal"         : "#a6cee3",
            "alkaline earth metal" : "#1f78b4",
            "metal"                : "#d93b43",
            "halogen"              : "#999d9a",
            "metalloid"            : "#e08d49",
            "noble gas"            : "#eaeaea",
            "nonmetal"             : "#f1d4Af",
            "transition metal"     : "#599d7A",
        }

        TOOLTIPS = [
            ("Name", "@name"),
            ("Atomic number", "@{atomic number}"),
            ("Atomic mass", "@{atomic mass}"),
            ("Type", "@metal"),
            ("CPK color", "$color[hex, swatch]:CPK"),
            ("Electronic configuration", "@{electronic configuration}"),
        ]

        p = figure(title="Periodic Table (omitting LA and AC Series)", width=1000, height=450,
                x_range=groups, y_range=list(reversed(periods)),
                tools="hover", toolbar_location=None, tooltips=TOOLTIPS)

        r = p.rect("group", "period", 0.95, 0.95, source=df, fill_alpha=0.6, legend_field="metal",
                color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))

        text_props = dict(source=df, text_align="left", text_baseline="middle")

        x = dodge("group", -0.4, range=p.x_range)

        p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)

        p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",
            text_font_size="11px", **text_props)

        p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name",
            text_font_size="7px", **text_props)

        p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass",
            text_font_size="7px", **text_props)

        p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")

        p.outline_line_color = None
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_standoff = 0
        #p.legend.orientation = "horizontal"
        #p.legend.location ="top_center"
        p.hover.renderers = [r] # only hover element boxes

        #show(p)
        st.bokeh_chart(p,use_container_width=True)


def main():
    st.title("Catalyst explorer from OC22")
    
    page = st.sidebar.selectbox("Choose a page", ["explorer","energy relationship","main"])
    mpr = MPRester("9wRdnFkL7BToI0le2f4G3UcS0W20tLwR") 

    if page == 'explorer':
        
        st.header("Periodic table")
        periodic_table()

        df_H = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/H_raw_data.csv')
        df_OH = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/OH_raw_data.csv')
        df_O = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/O_raw_data.csv')
        df = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/final_adss.csv')
        
        st.header("Filter OC22 dataset")
        st.dataframe(filter_dataframe(df))

        while True:
            target_id = st.number_input("Pick a mp-id:",0,10000000, value=19017)
            mp_id = "mp-"+str(target_id)
            doc = mpr.summary.search(material_ids=[mp_id])
            if doc is None:
                st.write("No data!!!")
            else:
                break
        
        st.write(doc[0].elements)

            
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header("DOS")
        dos = mpr.get_dos_by_material_id(mp_id)
        
        dos_plotter = DosPlotter()
        dos_plotter.add_dos_dict(dos.get_spd_dos())     
        st.pyplot(dos_plotter.show())
        
        # BS
        st.header("Bandstructure")
        bs = mpr.get_bandstructure_by_material_id(mp_id)
        bs_plotter = BSPlotter(bs)
        st.pyplot(bs_plotter.show())



    elif page=="energy relationship":
        df = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/final_adss.csv')
        

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)


        st.header("Adsorption energy relation")
        st.subheader("E_*H vs E_*O")
        fig, ax = plt.subplots()
        plt.scatter(x=df['ads_E_H'],y=df['ads_E_O'],c=df['ads_E_OH'], cmap='viridis')
        st.pyplot(fig)
        st.subheader("E_*H vs E_*OH")
        fig, ax = plt.subplots()
        plt.scatter(x=df['ads_E_H'],y=df['ads_E_OH'],c=df['ads_E_O'], cmap='viridis')
        st.pyplot(fig)
        st.subheader("E_*O vs E_*OH")
        fig, ax = plt.subplots()
        plt.scatter(x=df['ads_E_O'],y=df['ads_E_OH'],c=df['ads_E_H'], cmap='viridis')
        st.pyplot(fig)

    elif page == "main":
        df_H = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/H_raw_data.csv')
        df_OH = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/OH_raw_data.csv')
        df_O = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/O_raw_data.csv')

        H_ads_values = st.slider("pick H_ads", -10.0 , 10.0, (-10.0,10.0))
        O_ads_values = st.slider("pick O_ads", -10.0 , 10.0, (-10.0,10.0))
        OH_ads_values = st.slider("pick OH_ads", -10.0 , 10.0, (-10.0,10.0))
        st.write('H_ads range:', H_ads_values)
        st.write('H_ads_min: ', H_ads_values[0])
        st.write('H_ads_max: ', H_ads_values[1])

        st.title('H_short')
        df_H = pd.read_csv('https://raw.githubusercontent.com/flash-jaehyun/OCP_streamlit/main/data/H_raw_data.csv')
        #st.set_index = df[]

        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(df_H)

        st.dataframe(df_H)   


        #st.write(df_H.corr())

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df_H.corr(), annot=True, ax=ax)
        st.pyplot(fig)


if __name__ == "__main__":
    main()

