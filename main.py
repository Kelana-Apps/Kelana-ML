from datetime import datetime
from tsp import solve_tsp  # Import fungsi TSP
from cbf import recommend  # Import fungsi CBF
import pandas as pd

# Fungsi untuk menghitung durasi liburan dalam hari
def calculate_duration(start_date, end_date):
    start = datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.strptime(end_date, "%d-%m-%Y")
    return (end - start).days + 1  # Tambahkan 1 untuk menghitung hari terakhir

# Fungsi untuk mendapatkan rekomendasi per slot waktu
def get_recommendations_per_time_slot(city, price_category):
    time_slots = ['morning', 'afternoon', 'evening']
    recommendations_per_slot = {slot: pd.DataFrame() for slot in time_slots}

    for slot in time_slots:
        temp_recommendations = recommend(city, price_category, slot, top_n=3)  # Ambil rekomendasi terbaik untuk tiap waktu
        if isinstance(temp_recommendations, pd.DataFrame):
            recommendations_per_slot[slot] = temp_recommendations

    return recommendations_per_slot

# Fungsi untuk memilih satu tempat dari setiap slot waktu untuk setiap hari
def select_places_for_days(recommendations_per_slot, num_days):
    selected_places = []

    for day in range(num_days):
        day_places = {}
        for slot in recommendations_per_slot:
            if not recommendations_per_slot[slot].empty:
                selected_place = recommendations_per_slot[slot].iloc[0]  # Ambil tempat pertama dari setiap slot
                day_places[slot] = selected_place
                recommendations_per_slot[slot] = recommendations_per_slot[slot].iloc[1:]  # Hapus tempat yang sudah dipilih
        selected_places.append(day_places)

    return selected_places

# Fungsi utama untuk menjalankan aplikasi
def main():
    print("Selamat datang di Aplikasi Optimasi Itinerary!")
    print("Masukkan informasi berikut untuk memulai:\n")
    
    # Input dari pengguna
    city = input("Masukkan nama kota tujuan: ")
    start_date = input("Masukkan tanggal mulai liburan (dd-mm-yyyy): ")
    end_date = input("Masukkan tanggal selesai liburan (dd-mm-yyyy): ")
    price_category = input("Masukkan kategori harga (Murah/Sedang/Mahal): ")

    # Menghitung durasi liburan
    num_days = calculate_duration(start_date, end_date)
    print(f"Durasi liburan Anda adalah {num_days} hari.\n")

    # Mendapatkan rekomendasi per slot waktu
    recommendations_per_slot = get_recommendations_per_time_slot(city, price_category)

    # Memilih tempat untuk setiap hari
    selected_places = select_places_for_days(recommendations_per_slot, num_days)

    # Mengoptimalkan rute perjalanan
    total_distance = 0
    optimized_routes = []

    for day_idx, day_places in enumerate(selected_places):
        print(f"\nMengoptimalkan rute untuk Hari {day_idx + 1}...")
        
        places_with_coords_day = {
            place['Place_Name']: (float(place['Lat']), float(place['Long'])) 
            for slot, place in day_places.items()
        }

        # Menyelesaikan masalah TSP untuk rute perjalanan hari ini
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

    # Menyusun rencana perjalanan berdasarkan rute yang dioptimalkan
    itinerary = pd.DataFrame(columns=['Day', 'Time_Slot', 'Place_Name', 'Category', 'Description', 'Rating', 'Price', 'Coordinate', 'Opening_Time', 'Closing_Time'])
    
    # Tentukan rencana perjalanan berdasarkan rute yang dioptimalkan
    for day_idx, day_places in enumerate(selected_places):
        for slot, place in day_places.items():
            # Menambahkan data ke itinerary
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

    # Menampilkan rekomendasi tempat dan slot waktunya
    print("\nRekomendasi tempat wisata dan waktu kunjungan:")
    print(itinerary)

if __name__ == "__main__":
    main()