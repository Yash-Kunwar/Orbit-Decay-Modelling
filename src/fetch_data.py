import pandas as pd
from spacetrack import SpaceTrackClient
from time import sleep

USERNAME = 'kadhaiipaneer@gmail.com'
PASSWORD = 'workusingspacetrack'

# NORAD IDs of decayed objects -> (Tiangong-1, UARS, ROSAT, Phobos-Grunt, GOCE)
NORAD_IDS = [37820, 21701, 20638, 37872, 35243]

def fetch_space_track_data():
    print("logging into Space-Track...")
    
    # The client handles the session and authentication automatically
    st = SpaceTrackClient(identity=USERNAME, password=PASSWORD)
    
    all_tles = []
    
    for norad_id in NORAD_IDS:
        print(f"fetching last 100 TLEs for NORAD ID: {norad_id}...")
        
        try:
            # gp_history = General Perturbations (Historical data)
            # This safely builds the query and fetches the JSON
            data = st.gp_history(norad_cat_id=norad_id, orderby='EPOCH desc', limit=100)
            
            if data:
                for item in data:
                    all_tles.append({
                        'norad_id': item['NORAD_CAT_ID'],
                        'name': item['OBJECT_NAME'],
                        'epoch': item['EPOCH'],
                        'line1': item['TLE_LINE1'],
                        'line2': item['TLE_LINE2']
                    })
            else:
                print(f"no data was retrieved for {norad_id}.")
                
        except Exception as e:
            print(f"Error fetching data for {norad_id}: {e}")
            
        # Space-Track limits query speeds (max 30 requests per minute)
        sleep(2) 
        
    # Save to a CSV file
    if all_tles:
        df = pd.DataFrame(all_tles)
        df.to_csv('historical_tles.csv', index=False)
        print(f"\nsaved {len(df)} TLE records to historical_tles.csv")
    else:
        print("\nno data was retrieved across all IDs.")

if __name__ == "__main__":
    fetch_space_track_data()