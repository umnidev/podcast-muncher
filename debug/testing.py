import numpy as np
import sqlite3
import json


def dia_test():
    result = {'jobId': '02faaa54-beb5-4d21-8e42-80e3a7dcc150', 'status': 'succeeded', 'createdAt': '2025-07-30T20:49:39.730Z', 'updatedAt': '2025-07-30T20:50:19.087Z', 'output': {'diarization': [{'speaker': 'SPEAKER_01', 'start': 0.385, 'end': 5.225}, {'speaker': 'SPEAKER_01', 'start': 5.485, 'end': 6.565}, {'speaker': 'SPEAKER_02', 'start': 6.905, 'end': 7.285}, {'speaker': 'SPEAKER_02', 'start': 7.505, 'end': 12.585}, {'speaker': 'SPEAKER_02', 'start': 12.825, 'end': 13.105}, {'speaker': 'SPEAKER_00', 'start': 14.185, 'end': 19.825}, {'speaker': 'SPEAKER_00', 'start': 20.245, 'end': 25.145}, {'speaker': 'SPEAKER_02', 'start': 20.445, 'end': 20.465}, {'speaker': 'SPEAKER_02', 'start': 25.205, 'end': 28.945}, {'speaker': 'SPEAKER_01', 'start': 30.125, 'end': 40.165}, {'speaker': 'SPEAKER_01', 'start': 40.485, 'end': 44.325}, {'speaker': 'SPEAKER_01', 'start': 44.905, 'end': 53.385}, {'speaker': 'SPEAKER_01', 'start': 53.545, 'end': 58.585}, {'speaker': 'SPEAKER_01', 'start': 58.885, 'end': 60.485}, {'speaker': 'SPEAKER_02', 'start': 59.705, 'end': 60.325}, {'speaker': 'SPEAKER_02', 'start': 60.485, 'end': 61.805}, {'speaker': 'SPEAKER_01', 'start': 61.445, 'end': 65.545}, {'speaker': 'SPEAKER_02', 'start': 65.605, 'end': 68.425}, {'speaker': 'SPEAKER_01', 'start': 68.425, 'end': 71.605}, {'speaker': 'SPEAKER_01', 'start': 72.125, 'end': 78.085}, {'speaker': 'SPEAKER_00', 'start': 77.685, 'end': 82.725}, {'speaker': 'SPEAKER_00', 'start': 84.105, 'end': 85.305}, {'speaker': 'SPEAKER_01', 'start': 85.925, 'end': 91.505}, {'speaker': 'SPEAKER_00', 'start': 86.885, 'end': 87.565}, {'speaker': 'SPEAKER_01', 'start': 91.805, 'end': 93.085}, {'speaker': 'SPEAKER_00', 'start': 92.545, 'end': 99.185}, {'speaker': 'SPEAKER_01', 'start': 98.585, 'end': 102.585}, {'speaker': 'SPEAKER_00', 'start': 102.525, 'end': 103.005}, {'speaker': 'SPEAKER_01', 'start': 103.005, 'end': 108.765}, {'speaker': 'SPEAKER_00', 'start': 108.765, 'end': 108.785}, {'speaker': 'SPEAKER_01', 'start': 109.025, 'end': 112.065}, {'speaker': 'SPEAKER_01', 'start': 112.865, 'end': 114.485}, {'speaker': 'SPEAKER_01', 'start': 115.105, 'end': 131.865}, {'speaker': 'SPEAKER_00', 'start': 129.145, 'end': 129.185}, {'speaker': 'SPEAKER_00', 'start': 129.685, 'end': 130.025}, {'speaker': 'SPEAKER_00', 'start': 131.065, 'end': 132.205}, {'speaker': 'SPEAKER_01', 'start': 132.645, 'end': 133.505}, {'speaker': 'SPEAKER_00', 'start': 133.165, 'end': 137.585}, {'speaker': 'SPEAKER_01', 'start': 137.545, 'end': 137.825}, {'speaker': 'SPEAKER_00', 'start': 137.725, 'end': 141.385}, {'speaker': 'SPEAKER_01', 'start': 139.105, 'end': 139.785}, {'speaker': 'SPEAKER_01', 'start': 140.745, 'end': 153.565}, {'speaker': 'SPEAKER_00', 'start': 153.545, 'end': 153.885}, {'speaker': 'SPEAKER_01', 'start': 153.845, 'end': 155.745}, {'speaker': 'SPEAKER_00', 'start': 153.985, 'end': 154.025}, {'speaker': 'SPEAKER_00', 'start': 154.105, 'end': 157.725}, {'speaker': 'SPEAKER_01', 'start': 156.785, 'end': 162.065}, {'speaker': 'SPEAKER_01', 'start': 162.905, 'end': 171.185}, {'speaker': 'SPEAKER_01', 'start': 171.605, 'end': 179.365}, {'speaker': 'SPEAKER_01', 'start': 180.045, 'end': 196.585}, {'speaker': 'SPEAKER_02', 'start': 192.325, 'end': 194.265}, {'speaker': 'SPEAKER_00', 'start': 196.025, 'end': 197.805}, {'speaker': 'SPEAKER_01', 'start': 198.225, 'end': 198.525}, {'speaker': 'SPEAKER_00', 'start': 198.525, 'end': 199.385}, {'speaker': 'SPEAKER_01', 'start': 199.365, 'end': 202.945}, {'speaker': 'SPEAKER_00', 'start': 199.985, 'end': 200.645}, {'speaker': 'SPEAKER_00', 'start': 201.485, 'end': 202.385}, {'speaker': 'SPEAKER_02', 'start': 202.385, 'end': 202.465}, {'speaker': 'SPEAKER_00', 'start': 202.465, 'end': 202.505}, {'speaker': 'SPEAKER_02', 'start': 202.505, 'end': 202.785}, {'speaker': 'SPEAKER_00', 'start': 202.785, 'end': 202.845}, {'speaker': 'SPEAKER_00', 'start': 202.945, 'end': 203.605}, {'speaker': 'SPEAKER_01', 'start': 203.585, 'end': 209.245}, {'speaker': 'SPEAKER_01', 'start': 209.525, 'end': 211.125}, {'speaker': 'SPEAKER_00', 'start': 211.145, 'end': 211.405}, {'speaker': 'SPEAKER_02', 'start': 211.405, 'end': 211.865}, {'speaker': 'SPEAKER_01', 'start': 212.245, 'end': 212.685}, {'speaker': 'SPEAKER_02', 'start': 213.305, 'end': 216.705}, {'speaker': 'SPEAKER_02', 'start': 216.965, 'end': 218.905}, {'speaker': 'SPEAKER_00', 'start': 218.905, 'end': 218.925}, {'speaker': 'SPEAKER_02', 'start': 218.925, 'end': 219.325}, {'speaker': 'SPEAKER_00', 'start': 219.325, 'end': 219.345}, {'speaker': 'SPEAKER_02', 'start': 219.345, 'end': 219.365}, {'speaker': 'SPEAKER_00', 'start': 219.365, 'end': 219.585}, {'speaker': 'SPEAKER_01', 'start': 219.585, 'end': 219.605}, {'speaker': 'SPEAKER_00', 'start': 219.605, 'end': 219.765}, {'speaker': 'SPEAKER_00', 'start': 221.045, 'end': 221.085}, {'speaker': 'SPEAKER_01', 'start': 221.085, 'end': 226.225}, {'speaker': 'SPEAKER_00', 'start': 221.105, 'end': 221.765}, {'speaker': 'SPEAKER_01', 'start': 226.405, 'end': 232.985}, {'speaker': 'SPEAKER_01', 'start': 233.365, 'end': 236.125}, {'speaker': 'SPEAKER_01', 'start': 236.505, 'end': 238.625}, {'speaker': 'SPEAKER_00', 'start': 238.705, 'end': 243.825}, {'speaker': 'SPEAKER_00', 'start': 244.465, 'end': 254.465}, {'speaker': 'SPEAKER_00', 'start': 255.025, 'end': 261.325}, {'speaker': 'SPEAKER_00', 'start': 261.385, 'end': 263.525}, {'speaker': 'SPEAKER_01', 'start': 263.565, 'end': 264.005}, {'speaker': 'SPEAKER_00', 'start': 264.145, 'end': 264.745}, {'speaker': 'SPEAKER_01', 'start': 264.805, 'end': 266.405}, {'speaker': 'SPEAKER_00', 'start': 266.805, 'end': 274.465}, {'speaker': 'SPEAKER_01', 'start': 273.185, 'end': 274.165}, {'speaker': 'SPEAKER_01', 'start': 274.645, 'end': 276.465}, {'speaker': 'SPEAKER_00', 'start': 276.045, 'end': 279.945}, {'speaker': 'SPEAKER_01', 'start': 279.845, 'end': 280.165}, {'speaker': 'SPEAKER_00', 'start': 280.125, 'end': 280.505}, {'speaker': 'SPEAKER_01', 'start': 281.005, 'end': 282.645}, {'speaker': 'SPEAKER_00', 'start': 282.665, 'end': 284.065}, {'speaker': 'SPEAKER_01', 'start': 284.065, 'end': 286.565}, {'speaker': 'SPEAKER_00', 'start': 284.345, 'end': 290.425}, {'speaker': 'SPEAKER_02', 'start': 286.565, 'end': 286.645}, {'speaker': 'SPEAKER_02', 'start': 288.985, 'end': 290.525}, {'speaker': 'SPEAKER_02', 'start': 290.745, 'end': 293.985}, {'speaker': 'SPEAKER_00', 'start': 294.025, 'end': 306.945}, {'speaker': 'SPEAKER_02', 'start': 296.945, 'end': 297.805}, {'speaker': 'SPEAKER_02', 'start': 303.685, 'end': 303.705}, {'speaker': 'SPEAKER_00', 'start': 307.285, 'end': 309.325}, {'speaker': 'SPEAKER_02', 'start': 309.365, 'end': 310.505}, {'speaker': 'SPEAKER_00', 'start': 310.845, 'end': 316.665}, {'speaker': 'SPEAKER_01', 'start': 312.545, 'end': 312.565}, {'speaker': 'SPEAKER_02', 'start': 312.565, 'end': 312.825}, {'speaker': 'SPEAKER_01', 'start': 313.865, 'end': 315.125}, {'speaker': 'SPEAKER_01', 'start': 317.205, 'end': 317.325}, {'speaker': 'SPEAKER_00', 'start': 317.325, 'end': 322.785}, {'speaker': 'SPEAKER_01', 'start': 317.345, 'end': 318.765}, {'speaker': 'SPEAKER_01', 'start': 323.125, 'end': 337.785}, {'speaker': 'SPEAKER_00', 'start': 334.665, 'end': 335.565}, {'speaker': 'SPEAKER_01', 'start': 338.085, 'end': 340.445}, {'speaker': 'SPEAKER_01', 'start': 341.125, 'end': 342.845}, {'speaker': 'SPEAKER_00', 'start': 343.585, 'end': 356.825}, {'speaker': 'SPEAKER_01', 'start': 344.005, 'end': 344.365}, {'speaker': 'SPEAKER_00', 'start': 356.925, 'end': 357.425}, {'speaker': 'SPEAKER_01', 'start': 357.425, 'end': 357.445}, {'speaker': 'SPEAKER_01', 'start': 357.485, 'end': 357.505}, {'speaker': 'SPEAKER_00', 'start': 357.505, 'end': 366.565}, {'speaker': 'SPEAKER_01', 'start': 357.565, 'end': 358.225}, {'speaker': 'SPEAKER_02', 'start': 358.225, 'end': 358.245}, {'speaker': 'SPEAKER_01', 'start': 358.245, 'end': 358.345}, {'speaker': 'SPEAKER_02', 'start': 358.345, 'end': 358.365}, {'speaker': 'SPEAKER_02', 'start': 366.765, 'end': 367.105}, {'speaker': 'SPEAKER_00', 'start': 367.285, 'end': 377.205}, {'speaker': 'SPEAKER_02', 'start': 375.085, 'end': 375.605}, {'speaker': 'SPEAKER_02', 'start': 376.845, 'end': 398.325}, {'speaker': 'SPEAKER_00', 'start': 397.625, 'end': 402.045}, {'speaker': 'SPEAKER_02', 'start': 401.385, 'end': 459.845}, {'speaker': 'SPEAKER_02', 'start': 460.345, 'end': 527.605}, {'speaker': 'SPEAKER_01', 'start': 464.605, 'end': 464.845}, {'speaker': 'SPEAKER_01', 'start': 481.605, 'end': 481.885}, {'speaker': 'SPEAKER_01', 'start': 506.265, 'end': 506.325}, {'speaker': 'SPEAKER_00', 'start': 527.565, 'end': 536.645}, {'speaker': 'SPEAKER_02', 'start': 528.165, 'end': 528.925}, {'speaker': 'SPEAKER_02', 'start': 534.765, 'end': 582.265}, {'speaker': 'SPEAKER_00', 'start': 547.745, 'end': 547.965}, {'speaker': 'SPEAKER_01', 'start': 556.165, 'end': 556.485}, {'speaker': 'SPEAKER_01', 'start': 560.745, 'end': 561.085}, {'speaker': 'SPEAKER_02', 'start': 582.885, 'end': 590.425}, {'speaker': 'SPEAKER_00', 'start': 588.825, 'end': 590.765}, {'speaker': 'SPEAKER_02', 'start': 590.665, 'end': 592.865}, {'speaker': 'SPEAKER_02', 'start': 593.305, 'end': 599.985}, {'speaker': 'SPEAKER_00', 'start': 593.325, 'end': 593.625}]}}

    turns = result["output"]["diarization"]

    combined_turns = []
    prev_turn = None
    for turn in turns:
        print(f"{turn["speaker"]} start:{turn["start"]} end:{turn["end"]}")
        
        # first round
        if not prev_turn:
            prev_turn = turn
            start_of_combined_turn = turn["start"]
            continue

        # same speaker, continue
        if prev_turn["speaker"] == turn["speaker"]:
            prev_turn = turn
            continue

        # new speaker, record combined turn
        else:

            # store combined turn
            combined_turns.append({
                "end": prev_turn["end"],
                "speaker": prev_turn["speaker"],
                "start": start_of_combined_turn
            })

            # mark new start
            start_of_combined_turn = turn["start"]

            prev_turn = turn

    for combined_turn in combined_turns:
        print(f"c-> {combined_turn["speaker"]} start:{combined_turn["start"]} end:{combined_turn["end"]}")


def db_test():
    db = sqlite3.connect("test.db")
    cursor = db.cursor()
    try:
        cursor.execute(
                """
                CREATE TABLE transcript(
                    podcast_name,
                    episode_number,
                    transcript_json
                    );
                """
                )
        db.commit()
    except Exception as e:
        print(e)
        pass

    output = {'detected_language': 'no', 'segments': [{'end': 6.301, 'speaker': 'SPEAKER_01', 'start': 0.51, 'text': ' Jeg burde nesten introdusere deg som Norges nye YouTuber nå.', 'words': [{'end': 1.091, 'score': 0.743, 'speaker': 'SPEAKER_01', 'start': 0.51, 'word': 'Jeg'}, {'end': 1.272, 'score': 0.709, 'speaker': 'SPEAKER_01', 'start': 1.111, 'word': 'burde'}, {'end': 1.512, 'score': 0.805, 'speaker': 'SPEAKER_01', 'start': 1.312, 'word': 'nesten'}, {'end': 2.193, 'score': 0.454, 'speaker': 'SPEAKER_01', 'start': 1.552, 'word': 'introdusere'}, {'end': 2.334, 'score': 0, 'speaker': 'SPEAKER_01', 'start': 2.233, 'word': 'deg'}]}]}
    
    cursor.execute("INSERT INTO transcript(podcast_name, episode_number,transcript_json) VALUES(?, ?, ?)", ("wee", "614", json.dumps(output)))
    db.commit()

    cursor.execute("SELECT * FROM transcript")
    result = cursor.fetchall()
    print(result)



if __name__ == "__main__":
    # dia_test()
    db_test()