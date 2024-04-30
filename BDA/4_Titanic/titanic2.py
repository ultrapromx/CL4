import pandas as pd

def map_function(data):
    results = []
    # Processing for average age of deceased males
    if data['Sex'] == 'male' and data['Survived'] == 0 and pd.notnull(data['Age']):
        results.append(('avg_age_deceased_males', (data['Age'], 1)))  # (sum of ages, count)
    
    # Processing for deceased females in each class
    if data['Sex'] == 'female' and data['Survived'] == 0:
        results.append((f'deceased_females_class_{data["Pclass"]}', 1))  # count per class
    return results

def reduce_function(mapped_data):
    reduced_data = {}
    for key, value in mapped_data:
        if key.startswith('avg_age_deceased_males'):
            if key in reduced_data:
                reduced_data[key] = (reduced_data[key][0] + value[0], reduced_data[key][1] + value[1])
            else:
                reduced_data[key] = value
        else:
            if key in reduced_data:
                reduced_data[key] += value
            else:
                reduced_data[key] = value
    return reduced_data

def main(file_path):
    # Load the Titanic dataset
    titanic_data = pd.read_csv(file_path)

    mapped_results = []
    for _, row in titanic_data.iterrows():
        mapped_results.extend(map_function(row))
    
    reduced_results = reduce_function(mapped_results)
    
    # Calculate average age from the reduced results
    avg_age_key = 'avg_age_deceased_males'
    if avg_age_key in reduced_results and reduced_results[avg_age_key][1] > 0:
        average_age = reduced_results[avg_age_key][0] / reduced_results[avg_age_key][1]
        print(f"Average age of deceased males: {average_age:.2f} years")
    else:
        print("No deceased male data available to calculate average age.")
    
    # Print deceased females per class
    for key in reduced_results:
        if key.startswith('deceased_females_class'):
            print(f"Number of deceased females in {key.split('_')[-1]}: {reduced_results[key]}")

# Specify the path to the Titanic dataset
file_path = 'BDA\4_Titanic\Titanic-Dataset.csv'  # Update this path as needed
main(file_path)
