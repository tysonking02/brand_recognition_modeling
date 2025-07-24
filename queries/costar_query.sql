SELECT
    PropertyID AS property_id, 
    PropertyName AS property_name, 
    PropertyAddress AS property_address, 
    City AS city, 
    State AS state, 
    ZipCode AS zip_code, 
    Latitude AS latitude, 
    Longitude AS longitude, 
    MarketName AS market_name, 
    SubmarketName AS submarket_name, 
    UnitCount AS unit_count, 
    StarRating AS star_rating, 
    BuildingClass AS building_class, 
	Style as style,
    YearBuilt AS year_built, 
    YearRenovated AS year_renovated, 
    YEAR(LastSaleDate) AS year_acquired, 
    NumberOfStories AS number_of_stories,
    LandAreaSF AS land_area_sf, 
    TotalBuildings AS total_buildings, 
    CASE WHEN University IS NOT NULL THEN 1 ELSE 0 END AS is_university
FROM dbo.inv_FactCoStarProperty
where UnitCount >= 100 and AffordableType is NULL;
