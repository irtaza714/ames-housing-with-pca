from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
    MS_SubClass=int(request.form.get('MS_SubClass')),
    MS_Zoning=request.form.get('MS_Zoning'),
    Lot_Frontage=float(request.form.get('Lot_Frontage')),
    Lot_Area=int(request.form.get('Lot_Area')),
    Street=request.form.get('Street'),
    Lot_Shape=request.form.get('Lot_Shape'),
    Land_Contour=request.form.get('Land_Contour'),
    Lot_Config=request.form.get('Lot_Config'),
    Land_Slope=request.form.get('Land_Slope'),
    Neighborhood=request.form.get('Neighborhood'),
    Conition_One=request.form.get('Conition_One'),
    Condition_Two=request.form.get('Condition_Two'),
    Bldg_Type=request.form.get('Bldg_Type'),
    House_Style=request.form.get('House_Style'),
    Overall_Qual=int(request.form.get('Overall_Qual')),
    Overall_Cond=int(request.form.get('Overall_Cond')),
    Year_Built=int(request.form.get('Year_Built')),
    Year_Remod=int(request.form.get('Year_Remod')),
    Roof_Style=request.form.get('Roof_Style'),
    Roof_Matl=request.form.get('Roof_Matl'),
    Exterior_First=request.form.get('Exterior_First'),
    Exterior_Second=request.form.get('Exterior_Second'),
    Mas_Vnr_Type=request.form.get('Mas_Vnr_Type'),
    Mas_Vnr_Area=float(request.form.get('Mas_Vnr_Area')),
    Exter_Qual=request.form.get('Exter_Qual'),
    Exter_Cond=request.form.get('Exter_Cond'),
    Foundation=request.form.get('Foundation'),
    Bsmt_Qual=request.form.get('Bsmt_Qual'),
    Bsmt_Cond=request.form.get('Bsmt_Cond'),
    Bsmt_Exposure=request.form.get('Bsmt_Exposure'),
    BsmtFin_Type_One=request.form.get('BsmtFin_Type_One'),
    BsmtFin_SF_One=float(request.form.get('BsmtFin_SF_One')),
    BsmtFin_Type_Two=request.form.get('BsmtFin_Type_Two'),
    BsmtFin_SF_Two=float(request.form.get('BsmtFin_SF_Two')),
    Bsmt_Unf_SF=float(request.form.get('Bsmt_Unf_SF')),
    Total_Bsmt_SF=float(request.form.get('Total_Bsmt_SF')),
    Heating=request.form.get('Heating'),
    Heating_QC=request.form.get('Heating_QC'),
    Central_Air=request.form.get('Central_Air'),
    Electrical=request.form.get('Electrical'),
    First_Flr_SF=int(request.form.get('First_Flr_SF')),
    Second_Flr_SF=int(request.form.get('Second_Flr_SF')),
    Low_Qual_Fin_SF=int(request.form.get('Low_Qual_Fin_SF')),
    Gr_Liv_Area=int(request.form.get('Gr_Liv_Area')),
    Bsmt_Full_Bath=float(request.form.get('Bsmt_Full_Bath')),
    Bsmt_Half_Bath=float(request.form.get('Bsmt_Half_Bath')),
    Full_Bath=int(request.form.get('Full_Bath')),
    Half_Bath=int(request.form.get('Half_Bath')),
    Bedroom_AbvGr=int(request.form.get('Bedroom_AbvGr')),
    Kitchen_AbvGr=int(request.form.get('Kitchen_AbvGr')),
    Kitchen_Qual=request.form.get('Kitchen_Qual'),
    TotRms_AbvGrd=int(request.form.get('TotRms_AbvGrd')),
    Functional=request.form.get('Functional'),
    Fireplaces=int(request.form.get('Fireplaces')),
    Fireplace_Qu=request.form.get('Fireplace_Qu'),
    Garage_Type=request.form.get('Garage_Type'),
    Garage_Yr_Blt=float(request.form.get('Garage_Yr_Blt')),
    Garage_Finish=request.form.get('Garage_Finish'),
    Garage_Cars=float(request.form.get('Garage_Cars')),
    Garage_Area=float(request.form.get('Garage_Area')),
    Garage_Qual=request.form.get('Garage_Qual'),
    Garage_Cond=request.form.get('Garage_Cond'),
    Paved_Drive=request.form.get('Paved_Drive'),
    Wood_Deck_SF=int(request.form.get('Wood_Deck_SF')),
    Open_Porch_SF=int(request.form.get('Open_Porch_SF')),
    Enclosed_Porch=int(request.form.get('Enclosed_Porch')),
    Ssn_Porch=int(request.form.get('Ssn_Porch')),
    Screen_Porch=int(request.form.get('Screen_Porch')),
    Pool_Area=int(request.form.get('Pool_Area')),
    Misc_Val=int(request.form.get('Misc_Val')),
    Mo_Sold=int(request.form.get('Mo_Sold')),
    Yr_Sold=int(request.form.get('Yr_Sold')),
    Sale_Type=request.form.get('Sale_Type'),
    Sale_Condition=request.form.get('Sale_Condition')
    )
        pred_df = data.get_data_as_data_frame()
        print("Before Prediction", pred_df)

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)

        print("After Prediction", results[0])
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)