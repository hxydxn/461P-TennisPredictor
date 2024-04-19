import pandas as pd
import joblib

class ScorePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, player_shot_ball, shot_speed, player_1_position_x, player_1_position_y, player_2_position_x, player_2_position_y, player_serve):
        data = {
            'player_shot_ball': player_shot_ball,
            'shot_speed': shot_speed,
            'player_1_position_x': player_1_position_x,
            'player_1_position_y': player_1_position_y,
            'player_2_position_x': player_2_position_x,
            'player_2_position_y': player_2_position_y,
            'player_serve': player_serve
        }
        data_df = pd.DataFrame([data])
        return self.model.predict_proba(data_df)