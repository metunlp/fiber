# FIBER

## Dataset
**Dataset Name:** `FIBER - Factual Inference Bias Evaluation Resource`

**FIBER** is a high-quality dataset designed to evaluate **language-model factual inference bias** across three languages — **English (en)**, **Italian (it)**, and **Turkish (tr)**.

It contains both *single-entity* and *multi-entity* question–answer sets derived from structured world knowledge domains (e.g., capitals, car brands, time zones, official languages).

## Instructions on Running Scripts

### Step 1: Download Dependencies
The following Python libraries are required for evaluation:
- "torch"
- "transformers"
- "huggingface-hub"

To install them automatically, run the following command in your terminal: make requirements

### Step 2: Fill the Config Information
Provide the following before model tests in the config.json
1. Hugging Face token (hugging_face_token)
2. ID of the model you want to test (model_id)
3. Input directory (dataset_dir) (leave it as "dataset" if you have not changed the file structure)
4. Output directory (results_dir) ("results/MODEL_NAME" is suggested)

Example entries are provided below:
1. hugging_face_token : "YOUR_HUGGING_FACE_TOKEN"
2. model_id : "google/gemma-3-27b-it"
3. dataset_dir : "dataset"
4. output_dir : "results/gemma-3-27b"

In paths, don't leave any forward slashes at the end.

### Step 3: Run
Before running the script, make sure that you have downloaded dependencies and filled out the config information.

To start evaluation, run the following command in your terminal: make run

## Dataset Structure

The dataset is divided into two major parts:
```
.
├── multi_entity # Queries with multiple correct answers.
│   ├── car_brands
│   │   ├── car_brands_en_0.json
│   │   ├── car_brands_en_1.json
│   │   ├── car_brands_en_2.json
│   │   ├── car_brands_en_3.json
│   │   ├── car_brands_it_0.json
│   │   ├── car_brands_it_1.json
│   │   ├── car_brands_it_2.json
│   │   ├── car_brands_it_3.json
│   │   ├── car_brands_tr_0_0.json
│   │   ├── car_brands_tr_0_1.json
│   │   ├── car_brands_tr_1.json
│   │   ├── car_brands_tr_2.json
│   │   └── car_brands_tr_3.json
│   ├── countries_heritages
│   │   ├── countries_heritages_en_0.json
│   │   ├── countries_heritages_en_1.json
│   │   ├── countries_heritages_en_2.json
│   │   ├── countries_heritages_en_3.json
│   │   ├── countries_heritages_it_0.json
│   │   ├── countries_heritages_it_1.json
│   │   ├── countries_heritages_it_2.json
│   │   ├── countries_heritages_it_3.json
│   │   ├── countries_heritages_tr_0_0.json
│   │   ├── countries_heritages_tr_0_1.json
│   │   ├── countries_heritages_tr_1.json
│   │   ├── countries_heritages_tr_2.json
│   │   └── countries_heritages_tr_3.json
│   ├── countries_neighbors
│   │   ├── countries_neighbors_en_0.json
│   │   ├── countries_neighbors_en_1.json
│   │   ├── countries_neighbors_en_2.json
│   │   ├── countries_neighbors_en_3.json
│   │   ├── countries_neighbors_it_0.json
│   │   ├── countries_neighbors_it_1.json
│   │   ├── countries_neighbors_it_2.json
│   │   ├── countries_neighbors_it_3.json
│   │   ├── countries_neighbors_tr_0_0.json
│   │   ├── countries_neighbors_tr_0_1.json
│   │   ├── countries_neighbors_tr_1.json
│   │   ├── countries_neighbors_tr_2.json
│   │   └── countries_neighbors_tr_3.json
│   ├── countries_official_languages
│   │   ├── countries_official_languages_en_0.json
│   │   ├── countries_official_languages_en_1.json
│   │   ├── countries_official_languages_en_2.json
│   │   ├── countries_official_languages_en_3.json
│   │   ├── countries_official_languages_it_0.json
│   │   ├── countries_official_languages_it_1.json
│   │   ├── countries_official_languages_it_2.json
│   │   ├── countries_official_languages_it_3.json
│   │   ├── countries_official_languages_tr_0_0.json
│   │   ├── countries_official_languages_tr_0_1.json
│   │   ├── countries_official_languages_tr_1.json
│   │   ├── countries_official_languages_tr_2.json
│   │   └── countries_official_languages_tr_3.json
│   ├── countries_timezones
│   │   ├── countries_timezones_en_0.json
│   │   ├── countries_timezones_en_1.json
│   │   ├── countries_timezones_en_2.json
│   │   ├── countries_timezones_en_3.json
│   │   ├── countries_timezones_it_0.json
│   │   ├── countries_timezones_it_1.json
│   │   ├── countries_timezones_it_2.json
│   │   ├── countries_timezones_it_3.json
│   │   ├── countries_timezones_tr_0_0.json
│   │   ├── countries_timezones_tr_0_1.json
│   │   ├── countries_timezones_tr_1.json
│   │   ├── countries_timezones_tr_2.json
│   │   └── countries_timezones_tr_3.json
│   ├── mobile_network_operators
│   │   ├── mobile_network_operators_en_0.json
│   │   ├── mobile_network_operators_en_1.json
│   │   ├── mobile_network_operators_en_2.json
│   │   ├── mobile_network_operators_en_3.json
│   │   ├── mobile_network_operators_it_0.json
│   │   ├── mobile_network_operators_it_1.json
│   │   ├── mobile_network_operators_it_2.json
│   │   ├── mobile_network_operators_it_3.json
│   │   ├── mobile_network_operators_tr_0_0.json
│   │   ├── mobile_network_operators_tr_0_1.json
│   │   ├── mobile_network_operators_tr_1.json
│   │   ├── mobile_network_operators_tr_2.json
│   │   └── mobile_network_operators_tr_3.json
│   ├── polyglot_celebs
│   │   ├── polyglot_celebs_en_0.json
│   │   ├── polyglot_celebs_en_1.json
│   │   ├── polyglot_celebs_en_2.json
│   │   ├── polyglot_celebs_en_3.json
│   │   ├── polyglot_celebs_it_0.json
│   │   ├── polyglot_celebs_it_1.json
│   │   ├── polyglot_celebs_it_2.json
│   │   ├── polyglot_celebs_it_3.json
│   │   ├── polyglot_celebs_tr_0_0.json
│   │   ├── polyglot_celebs_tr_0_1.json
│   │   ├── polyglot_celebs_tr_1.json
│   │   ├── polyglot_celebs_tr_2.json
│   │   └── polyglot_celebs_tr_3.json
│   └── top_500_universities
│       ├── top_500_universities_en_0.json
│       ├── top_500_universities_en_1.json
│       ├── top_500_universities_en_2.json
│       ├── top_500_universities_en_3.json
│       ├── top_500_universities_it_0.json
│       ├── top_500_universities_it_1.json
│       ├── top_500_universities_it_2.json
│       ├── top_500_universities_it_3.json
│       ├── top_500_universities_tr_0_0.json
│       ├── top_500_universities_tr_0_1.json
│       ├── top_500_universities_tr_1.json
│       ├── top_500_universities_tr_2.json
│       └── top_500_universities_tr_3.json
└── single_entity # Queries with a single correct answer.
    ├── atomic_numbers
    │   ├── atomic_numbers_en_0.json
    │   ├── atomic_numbers_en_1.json
    │   ├── atomic_numbers_it_0.json
    │   ├── atomic_numbers_it_1.json
    │   ├── atomic_numbers_tr_0_0.json
    │   ├── atomic_numbers_tr_0_1.json
    │   └── atomic_numbers_tr_1.json
    ├── capital_cities
    │   ├── capital_cities_en_0.json
    │   ├── capital_cities_en_1.json
    │   ├── capital_cities_it_0.json
    │   ├── capital_cities_it_1.json
    │   ├── capital_cities_tr_0_0.json
    │   ├── capital_cities_tr_0_1.json
    │   └── capital_cities_tr_1.json
    ├── ccTLD
    │   ├── ccTLD_en_0.json
    │   ├── ccTLD_en_1.json
    │   ├── ccTLD_it_0.json
    │   ├── ccTLD_it_1.json
    │   ├── ccTLD_tr_0_0.json
    │   ├── ccTLD_tr_0_1.json
    │   └── ccTLD_tr_1.json
    ├── chemical_symbols
    │   ├── chemical_symbols_en_0.json
    │   ├── chemical_symbols_en_1.json
    │   ├── chemical_symbols_it_0.json
    │   ├── chemical_symbols_it_1.json
    │   ├── chemical_symbols_tr_0_0.json
    │   ├── chemical_symbols_tr_0_1.json
    │   └── chemical_symbols_tr_1.json
    ├── founding_locations
    │   ├── founding_locations_en_0.json
    │   ├── founding_locations_en_1.json
    │   ├── founding_locations_it_0.json
    │   ├── founding_locations_it_1.json
    │   ├── founding_locations_tr_0_0.json
    │   ├── founding_locations_tr_0_1.json
    │   └── founding_locations_tr_1.json
    ├── locations_of_sites
    │   ├── locations_of_sites_en_0.json
    │   ├── locations_of_sites_en_1.json
    │   ├── locations_of_sites_it_0.json
    │   ├── locations_of_sites_it_1.json
    │   ├── locations_of_sites_tr_0_0.json
    │   ├── locations_of_sites_tr_0_1.json
    │   └── locations_of_sites_tr_1.json
    ├── original_langs_of_books
    │   ├── original_langs_of_books_en_0.json
    │   ├── original_langs_of_books_en_1.json
    │   ├── original_langs_of_books_it_0.json
    │   ├── original_langs_of_books_it_1.json
    │   ├── original_langs_of_books_tr_0_0.json
    │   ├── original_langs_of_books_tr_0_1.json
    │   └── original_langs_of_books_tr_1.json
    └── product_maker
        ├── product_maker_en_0.json
        ├── product_maker_en_1.json
        ├── product_maker_it_0.json
        ├── product_maker_it_1.json
        ├── product_maker_tr_0_0.json
        ├── product_maker_tr_0_1.json
        └── product_maker_tr_1.json
```

### File Naming Convention

Each file follows the structure:

| Component | Example | Meaning |
|------------|----------|---------|
| `<topic>` | `countries_official_languages` | Knowledge domain |
| `<language>` | `en`, `tr`, `it` | Dataset language |
| `<index>` | `0`, `1`, `2`, `3`, or `0_0`, `0_1` | Split or subset index |

Turkish (`_tr_`) topics include subsets (`_0_0`, `_0_1`) for grammar purposes.