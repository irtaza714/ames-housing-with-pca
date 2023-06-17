import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            pca_path = "artifacts/pca.pkl"
            pca_components = "artifacts/pca_components.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            pca = load_object(file_path=pca_path)
            data_scaled = preprocessor.transform(features)
            data_scaled_df = pd.DataFrame(data_scaled)
            for i in range(len(data_scaled_df.columns)):  
                data_scaled_df = data_scaled_df.rename(columns={data_scaled_df.columns[i]: f'c{i+1}'})
            
            pca.n_components = pca_components
            data_scaled_df_pca = pca.transform (data_scaled_df)

            preds = model.predict(data_scaled_df_pca)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, MS_SubClass: int, MS_Zoning: str, Lot_Frontage: float, Lot_Area: int,
       Street: str, Lot_Shape: str, Land_Contour: str, Utilities: str,
       Lot_Config: str, Land_Slope: str, Neighborhood: str, Conition_One: str,
       Condition_Two: str, Bldg_Type: str, House_Style: str, Overall_Qual: int,
       Overall_Cond: int, Year_Built: int, Year_Remod: int, Roof_Style: str, Roof_Matl: str,
       Exterior_First: str, Exterior_Second: str, Mas_Vnr_Type: str, Mas_Vnr_Area: float,
       Exter_Qual: str, Exter_Cond: str, Foundation: str, Bsmt_Qual: str, Bsmt_Cond: str,
       Bsmt_Exposure: str, BsmtFin_Type_One: str, BsmtFin_SF_One: float,
       BsmtFin_Type_Two: str, BsmtFin_SF_Two: str, Bsmt_Unf_SF: float, Total_Bsmt_SF: float,
       Heating: str, Heating_QC: str, Central_Air: str, Electrical: str, First_Flr_SF: int,
       Second_Flr_SF: int, Low_Qual_Fin_SF: int, Gr_Liv_Area: int, Bsmt_Full_Bath: float,
       Bsmt_Half_Bath: float, Full_Bath: int, Half_Bath: int, Bedroom_AbvGr: int,
       Kitchen_AbvGr: int, Kitchen_Qual: str, TotRms_AbvGrd: int, Functional: str,
       Fireplaces: int, Fireplace_Qu: str, Garage_Type: str, Garage_Yr_Blt: float,
       Garage_Finish: str, Garage_Cars: float, Garage_Area: float, Garage_Qual: str,
       Garage_Cond: str, Paved_Drive: str, Wood_Deck_SF: int, Open_Porch_SF: int,
       Enclosed_Porch: int, Ssn_Porch: int, Screen_Porch: int, Pool_Area: int, Misc_Val: int,
       Mo_Sold: int, Yr_Sold: int, Sale_Type: str, Sale_Condition: str):
        
        self.MS_SubClass = MS_SubClass
        self.MS_Zoning = MS_Zoning
        self.Lot_Frontage = Lot_Frontage
        self.Lot_Area = Lot_Area
        self.Street = Street
        self.Lot_Shape = Lot_Shape
        self.Land_Contour = Land_Contour
        self.Utilities = Utilities
        self.Lot_Config = Lot_Config
        self.Land_Slope = Land_Slope
        self.Neighborhood = Neighborhood
        self.Conition_One = Conition_One
        self.Condition_Two = Condition_Two
        self.Bldg_Type = Bldg_Type
        self.House_Style = House_Style
        self.Overall_Qual = Overall_Qual
        self.Overall_Cond = Overall_Cond
        self.Year_Built = Year_Built
        self.Year_Remod = Year_Remod
        self.Roof_Style = Roof_Style
        self.Roof_Matl = Roof_Matl
        self.Exterior_First = Exterior_First
        self.Exterior_Second = Exterior_Second
        self.Mas_Vnr_Type = Mas_Vnr_Type
        self.Mas_Vnr_Area = Mas_Vnr_Area
        self.Exter_Qual = Exter_Qual
        self.Exter_Cond = Exter_Cond
        self.Foundation = Foundation
        self.Bsmt_Qual = Bsmt_Qual
        self.Bsmt_Cond = Bsmt_Cond
        self.Bsmt_Exposure = Bsmt_Exposure
        self.BsmtFin_Type_One = BsmtFin_Type_One
        self.BsmtFin_SF_One = BsmtFin_SF_One
        self.BsmtFin_Type_Two = BsmtFin_Type_Two
        self.BsmtFin_SF_Two = BsmtFin_SF_Two
        self.Bsmt_Unf_SF = Bsmt_Unf_SF
        self.Total_Bsmt_SF = Total_Bsmt_SF
        self.Heating = Heating
        self.Heating_QC = Heating_QC
        self.Central_Air = Central_Air
        self.Electrical = Electrical
        self.First_Flr_SF = First_Flr_SF
        self.Second_Flr_SF = Second_Flr_SF
        self.Low_Qual_Fin_SF = Low_Qual_Fin_SF
        self.Gr_Liv_Area = Gr_Liv_Area
        self.Bsmt_Full_Bath = Bsmt_Full_Bath
        self.Bsmt_Half_Bath = Bsmt_Half_Bath
        self.Full_Bath = Full_Bath
        self.Half_Bath = Half_Bath
        self.Bedroom_AbvGr = Bedroom_AbvGr
        self.Kitchen_AbvGr = Kitchen_AbvGr
        self.Kitchen_Qual = Kitchen_Qual
        self.TotRms_AbvGrd = TotRms_AbvGrd
        self.Functional = Functional
        self.Fireplaces = Fireplaces
        self.Fireplace_Qu = Fireplace_Qu
        self.Garage_Type = Garage_Type
        self.Garage_Yr_Blt = Garage_Yr_Blt
        self.Garage_Finish = Garage_Finish
        self.Garage_Cars = Garage_Cars
        self.Garage_Area = Garage_Area
        self.Garage_Qual = Garage_Qual
        self.Garage_Cond = Garage_Cond
        self.Paved_Drive = Paved_Drive
        self.Wood_Deck_SF = Wood_Deck_SF
        self.Open_Porch_SF = Open_Porch_SF
        self.Enclosed_Porch = Enclosed_Porch
        self.Ssn_Porch = Ssn_Porch
        self.Screen_Porch = Screen_Porch
        self.Pool_Area = Pool_Area
        self.Misc_Val = Misc_Val
        self.Mo_Sold = Mo_Sold
        self.Yr_Sold = Yr_Sold
        self.Sale_Type = Sale_Type
        self.Sale_Condition = Sale_Condition
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "MS_SubClass": [self.MS_SubClass],
                "MS_Zoning": [self.MS_Zoning],
                "Lot_Frontage" : [self.Lot_Frontage], 
                "Lot_Area" : [self.Lot_Area],
                "Street" : [self.Street],
                "Lot_Shape" : [self.Lot_Shape], 
                "Land_Contour" : [self.Land_Contour],
                "Utilities" : [self.Utilities],
                "Lot_Config" : [self.Lot_Config],
                "Land_Slope" : [self.Land_Slope],
                "Neighborhood" : [self.Neighborhood],
                "Conition_One" : [self.Conition_One],
                "Condition_Two" : [self.Condition_Two],
                "Bldg_Type" : [self.Bldg_Type],
                "House_Style" : [self.House_Style],
                "Overall_Qual" : [self.Overall_Qual],
                "Overall_Cond" : [self.Overall_Cond],
                "Year_Built" : [self.Year_Built],
                "Year_Remod" : [self.Year_Remod],
                "Roof_Style" : [self.Roof_Style],
                "Roof_Matl" : [self.Roof_Matl],
                "Exterior_First" : [self.Exterior_First],
                "Exterior_Second" : [self.Exterior_Second],
                "Mas_Vnr_Type" : [self.Mas_Vnr_Type],
                "Mas_Vnr_Area" : [self.Mas_Vnr_Area],
                "Exter_Qual" : [self.Exter_Qual],
                "Exter_Cond" : [self.Exter_Cond],
                "Foundation" : [self.Foundation],
                "Bsmt_Qual" : [self.Bsmt_Qual],
                "Bsmt_Cond" : [self.Bsmt_Cond],
                "Bsmt_Exposure" : [self.Bsmt_Exposure],
                "BsmtFin_Type_One" : [self.BsmtFin_Type_One],
                "BsmtFin_SF_One" : [self.BsmtFin_SF_One],
                "BsmtFin_Type_Two" : [self.BsmtFin_Type_Two],
                "BsmtFin_SF_Two" : [self.BsmtFin_SF_Two],
                "Bsmt_Unf_SF" : [self.Bsmt_Unf_SF],
                "Total_Bsmt_SF" : [self.Total_Bsmt_SF],
                "Heating" : [self.Heating],
                "Heating_QC" : [self.Heating_QC],
                "Central_Air" : [self.Central_Air],
                "Electrical" : [self.Electrical],
                "First_Flr_SF" : [self.First_Flr_SF],
                "Second_Flr_SF" : [self.Second_Flr_SF],
                "Low_Qual_Fin_SF" : [self.Low_Qual_Fin_SF],
                "Gr_Liv_Area" : [self.Gr_Liv_Area],
                "Bsmt_Full_Bath" : [self.Bsmt_Full_Bath],
                "Bsmt_Half_Bath" : [self.Bsmt_Half_Bath],
                "Full_Bath" : [self.Full_Bath],
                "Half_Bath" : [self.Half_Bath],
                "Bedroom_AbvGr" : [self.Bedroom_AbvGr],
                "Kitchen_AbvGr" : [self.Kitchen_AbvGr],
                "Kitchen_Qual" : [self.Kitchen_Qual],
                "TotRms_AbvGrd" : [self.TotRms_AbvGrd],
                "Functional" : [self.Functional],
                "Fireplaces" : [self.Fireplaces],
                "Fireplace_Qu" : [self.Fireplace_Qu],
                "Garage_Type" : [self.Garage_Type],
                "Garage_Yr_Blt" : [self.Garage_Yr_Blt],
                "Garage_Finish" : [self.Garage_Finish],
                "Garage_Cars" : [self.Garage_Cars],
                "Garage_Area" : [self.Garage_Area],
                "Garage_Qual" : [self.Garage_Qual],
                "Garage_Cond" : [self.Garage_Cond],
                "Paved_Drive" : [self.Paved_Drive],
                "Wood_Deck_SF" : [self.Wood_Deck_SF],
                "Open_Porch_SF" : [self.Open_Porch_SF],
                "Enclosed_Porch" : [self.Enclosed_Porch],
                "Ssn_Porch" : [self.Ssn_Porch],
                "Screen_Porch" : [self.Screen_Porch],
                "Pool_Area" : [self.Pool_Area],
                "Misc_Val" : [self.Misc_Val],
                "Mo_Sold" : [self.Mo_Sold],
                "Yr_Sold" : [self.Yr_Sold],
                "Sale_Type" : [self.Sale_Type],
                "Sale_Condition" : [self.Sale_Condition]
               }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)