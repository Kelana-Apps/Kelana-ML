from datetime import datetime
from tsp import solve_tsp
from cbf import recommend
import pandas as pd

# Function to calculate vacation duration
def calculate_duration(start_date, end_date):
    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    return (end - start).days + 1  # Add 1 to count the last day

# Function to get recommendations per time slot
def get_recommendations_per_time_slot(city, price_category):
    time_slots = ['morning', 'afternoon', 'evening']
    recommendations_per_slot = {slot: pd.DataFrame() for slot in time_slots}

    for slot in time_slots:
        temp_recommendations = recommend(city, price_category, slot, top_n=3)  # Get the best recommendation for each time slot
        if isinstance(temp_recommendations, pd.DataFrame):
            recommendations_per_slot[slot] = temp_recommendations

    return recommendations_per_slot

# Function to select one place from each time slot for each day
def select_places_for_days(recommendations_per_slot, num_days):
    selected_places = []

    for day in range(num_days):
        day_places = {}
        for slot in recommendations_per_slot:
            if not recommendations_per_slot[slot].empty:
                selected_place = recommendations_per_slot[slot].iloc[0]  # Get the first place from each time slot
                day_places[slot] = selected_place
                recommendations_per_slot[slot] = recommendations_per_slot[slot].iloc[1:]  # Remove the selected place
        selected_places.append(day_places)

    return selected_places

# The main function to run the application
def main():
    print("Selamat datang di Aplikasi Optimasi Itinerary!")
    print("Masukkan informasi berikut untuk memulai:\n")
    
    # Input user
    city = input("Masukkan nama kota tujuan: ")
    start_date = input("Masukkan tanggal mulai liburan (dd-mm-yyyy): ")
    end_date = input("Masukkan tanggal selesai liburan (dd-mm-yyyy): ")
    price_category = input("Masukkan kategori harga (Murah/Sedang/Mahal): ")

    # Calculate vacation duration
    num_days = calculate_duration(start_date, end_date)
    print(f"Durasi liburan Anda adalah {num_days} hari.\n")

    # Get recommendations per time slot
    recommendations_per_slot = get_recommendations_per_time_slot(city, price_category)

    # Select one place from each time slot for each day
    selected_places = select_places_for_days(recommendations_per_slot, num_days)

    # Optimize TSP for each day
    total_distance = 0
    optimized_routes = []

    for day_idx, day_places in enumerate(selected_places):
        print(f"\nMengoptimalkan rute untuk Hari {day_idx + 1}...")
        
        places_with_coords_day = {
            place['Place_Name']: (float(place['Lat']), float(place['Long'])) 
            for slot, place in day_places.items()
        }

        # Solve the TSP problem for today's travel route
        route_info = solve_tsp(places_with_coords_day)

        if route_info:
            optimized_route = route_info['route']
            day_distance = route_info['total_distance']
            optimized_routes.append(optimized_route)
            total_distance += day_distance
            print("\nRute perjalanan yang disarankan:")
            for idx, place in enumerate(optimized_route):
                print(f"{idx + 1}. {place}")
            print(f"\nTotal jarak perjalanan untuk hari ini: {day_distance:.2f} km")
        else:
            print("Gagal mengoptimalkan rute perjalanan untuk hari ini.")

    print(f"\nTotal jarak perjalanan untuk {num_days} hari: {total_distance:.2f} km")

    # Prepare the travel plan based on the optimized route
    itinerary = pd.DataFrame(columns=['Day', 'Time_Slot', 'Place_Name', 'Category', 'Description', 'Rating', 'Price', 'Coordinate', 'Opening_Time', 'Closing_Time'])
    
    # Determine the travel plan based on the optimized route
    for day_idx, day_places in enumerate(selected_places):
        for slot, place in day_places.items():
            # Add the selected place to the itinerary
            itinerary = pd.concat([itinerary, pd.DataFrame([{
                'Day': day_idx + 1,
                'Time_Slot': slot,
                'Place_Name': place['Place_Name'],
                'Category': place['Category'],
                'Description': place['Description'],
                'Rating': place['Rating'],
                'Price': place['Price'],
                'Coordinate': (place['Lat'], place['Long']),
                'Opening_Time': place['Opening_Time'],
                'Closing_Time': place['Closing_Time']
            }])], ignore_index=True)

    # Print the travel plan
    print("\nRekomendasi tempat wisata dan waktu kunjungan:")
    print(itinerary)

if __name__ == "__main__":
    main()