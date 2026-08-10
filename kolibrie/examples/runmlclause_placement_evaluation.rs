use kolibrie::sparql_database::SparqlDatabase;
use kolibrie::streamertail_optimizer::operators::{LogicalOperator, PhysicalOperator};
use kolibrie::parser::parse_sparql_query;
use kolibrie::streamertail_optimizer::utils::*;
use kolibrie::execute_query::execute_query_rayon_parallel2_volcano;
use kolibrie::execute_query::process_variables;
use kolibrie::streamertail_optimizer::optimizer::Streamertail;
// use polars::prelude::CloudScheme::File;
use polars::prelude::*;
use std::error::Error;
use std::fmt::format;
use std::fs::File;
use std::time::Instant;

fn append_alarm_fanout_triples(data: &mut String, machine_id: usize) {
    for alarm_idx in 0..3 {
        let alarm_uri = format!(
            r#"<http://example.org#machine{}> <http://example.org#hasAlarm> <http://example.org#alarm{}_{}> ."#,
            machine_id, machine_id, alarm_idx
        );
        let prediction_activation = format!(
            r#""1" <http://example.org#activates> <http://example.org#alarm{}_{}> ."#,
            machine_id, alarm_idx
        );
        data.push_str(&format!("\n{}\n{}", alarm_uri, prediction_activation));

        for action_idx in 0..2 {
            let action_uri = format!(
                r#"<http://example.org#alarm{}_{}> <http://example.org#activationType> <http://example.org#action{}_{}_{}> ."#,
                machine_id, alarm_idx, machine_id, alarm_idx, action_idx
            );
            let support_uri = format!(
                r#"<http://example.org#action{}_{}_{}> <http://example.org#supportedBy> <http://example.org#machine{}> ."#,
                machine_id, alarm_idx, action_idx, machine_id
            );
            for team_id in 1..10{
                let inform_uri = format!(
                r#"<http://example.org#action{}_{}_{}> <http://example.org#informs> <http://example.org#team{}_{}> ."#,
                machine_id, alarm_idx, action_idx, machine_id, team_id);
                let team_maintains = format!(
                    r#"<http://example.org#team{}_{}> <http://example.org#maintains> <http://example.org#machine{}> ."#,
                    machine_id, team_id, machine_id);
                data.push_str(&format!("\n{}\n{}\n{}\n{}", action_uri, support_uri, inform_uri, team_maintains));
            }
        }
    }
}

fn generate_data_1000() -> Result<String, PolarsError> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("examples/smart_manufacturing_data.csv");
    let file = File::open(&path).map_err(|e| PolarsError::ComputeError("oops1".into()))?;
    let df = CsvReader::new(file)
    .with_options(CsvReadOptions::default().with_has_header(true)).finish()?;

    // 2. Fetch the elements of a row by its index (e.g., first row = index 0)
    let row_index = 0;

    let mut data_1000 = r#"@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix fac: <https://factory.com#> .
    @prefix mls: <http://www.w3.org/ns/mls#> .
    @prefix ext: <http://www.thesisextension.org/runml#> .
    fac:mlruntime1 rdf:type mls:Run .
    fac:mlruntime1 mls:hasOutput ext:model1 .
    ext:model1 ext:hasModelName "model1" .
    ext:model1 rdf:type mls:Model .
    fac:mlruntime2 rdf:type mls:Run .
    fac:mlruntime2 mls:hasOutput ext:model2 .
    ext:model2 ext:hasModelName "model2" .
    "model2" rdf:type mls:Model .
    fac:mlruntime3 rdf:type mls:Run .
    fac:mlruntime3 mls:hasOutput ext:model3 .
    ext:model3 ext:hasModelName "model3" .
    ext:model3 rdf:type mls:Model ."#.to_string();
    for row in 0..1000 {
        append_alarm_fanout_triples(&mut data_1000, row);
        let row_elem = df.get_row(row)?;
        if let Some(temp) = row_elem.0.get(2){
            let tempStr1 = format!(r#"<http://example.org#machine{}> rdf:type <http://example.org#Room> .
            "#, row + 1);
            let tempStr2 = format!(r#"<http://example.org#machine{}> <http://example.org#tempValue> "{}" .
            "#, row + 1, temp);
            if let Some(vibr) = row_elem.0.get(3){
                let vibrStr = format!(r#"<http://example.org#machine{}> <http://example.org#vibrValue> "{}" .
                "#, row + 1, vibr);
                if let Some(humid) = row_elem.0.get(4){
                    let humidStr = format!(r#"<http://example.org#machine{}> <http://example.org#humidValue> "{}" .
                    "#, row + 1, humid);
                    if let Some(energy) = row_elem.0.get(6){
                        let energyStr = format!(r#"<http://example.org#machine{}> <http://example.org#powerValue> "{}" .
                        "#, row + 1, energy);
                        data_1000 = format!("{}{}{}{}{}{}", data_1000, tempStr1, tempStr2, vibrStr, humidStr, energyStr);
                    }
                }
            }
        }
    }
    for job in 1..10 {
        for machine in 0..1000 {
            let jobStr = format!(r#"<http://example.org#job{}> <http://example.org#hasAccess> <http://example.org#machine{}> .
            "#, job, machine);
            let jobStr2 = format!(r#"<http://example.org#job{}> <http://example.org#hasSalary> "{}000" .
            "#, job, job);
            data_1000 = format!("{}{}{}", data_1000, jobStr, jobStr2);
        }

        for person in 0..5 {
            let perStr = format!(r#"<http://example.org#person{}{}> <http://example.org#works> <http://example.org#job{}> .
            "#, person, job, job);
            let perStr2 = format!(r#"<http://example.org#person{}{}> <http://example.org#hasPet> <http://example.org#dog> .
            "#, person, job);
            data_1000 = format!("{}{}{}", data_1000, perStr, perStr2);
        }
        for person in 5..10 {
            let perStr = format!(r#"<http://example.org#person{}{}> <http://example.org#works> <http://example.org#job{}> .
            "#, person, job, job);
            let perStr2 = format!(r#"<http://example.org#person{}{}> <http://example.org#hasPet> <http://example.org#cat> .
            "#, person, job);
            data_1000 = format!("{}{}{}", data_1000, perStr, perStr2);
        }
    }

    return Ok(data_1000);
}

fn generate_data(data: &String, numMachines: usize) -> Result<String, PolarsError> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("examples/smart_manufacturing_data.csv");
    let file = File::open(&path).map_err(|e| PolarsError::ComputeError("File cannot be opened".into()))?;
    let df = CsvReader::new(file)
    .with_options(CsvReadOptions::default().with_has_header(true)).finish()?;

    // 2. Fetch the elements of a row by its index (e.g., first row = index 0)
    let row_index = 0;

    let mut data_n = data.clone();
    for row in (numMachines - 1000)..numMachines {
        append_alarm_fanout_triples(&mut data_n, row);
        let row_elem = df.get_row(row)?;
        if let Some(temp) = row_elem.0.get(2){
            let tempStr1 = format!(r#"<http://example.org#machine{}> rdf:type <http://example.org#Room> .
            "#, row + 1);
            let tempStr2 = format!(r#"<http://example.org#machine{}> <http://example.org#tempValue> "{}" .
            "#, row + 1, temp);
            if let Some(vibr) = row_elem.0.get(3){
                let vibrStr = format!(r#"<http://example.org#machine{}> <http://example.org#vibrValue> "{}" .
                "#, row + 1, vibr);
                if let Some(humid) = row_elem.0.get(4){
                    let humidStr = format!(r#"<http://example.org#machine{}> <http://example.org#humidValue> "{}" .
                    "#, row + 1, humid);
                    if let Some(energy) = row_elem.0.get(6){
                        let energyStr = format!(r#"<http://example.org#machine{}> <http://example.org#powerValue> "{}" .
                        "#, row + 1, energy);
                        data_n = format!("{}{}{}{}{}{}", data_n, tempStr1, tempStr2, vibrStr, humidStr, energyStr);
                    }
                }
            }
        }
    }
    for job in 1..10 {
        for machine in (numMachines - 1000)..numMachines {
            let jobStr = format!(r#"<http://example.org#job{}> <http://example.org#hasAccess> <http://example.org#machine{}> .
            "#, job, machine);
            let jobStr2 = format!(r#"<http://example.org#job{}> <http://example.org#hasSalary> "{}000" .
            "#, job, job);
            data_n = format!("{}{}{}", data_n, jobStr, jobStr2);
        }

        for person in 1..5 {
            let perStr = format!(r#"<http://example.org#person{}{}> <http://example.org#works> <http://example.org#job{}> .
            "#, person, job, job);
            let perStr2 = format!(r#"<http://example.org#person{}{}> <http://example.org#hasPet> <http://example.org#dog> .
            "#, person, job);
            data_n = format!("{}{}{}", data_n, perStr, perStr2);
        }
        for person in 5..10 {
            let perStr = format!(r#"<http://example.org#person{}{}> <http://example.org#works> <http://example.org#job{}> .
            "#, person, job, job);
            let perStr2 = format!(r#"<http://example.org#person{}{}> <http://example.org#hasPet> <http://example.org#cat> .
            "#, person, job);
            data_n = format!("{}{}{}", data_n, perStr, perStr2);
        }
    }

    return Ok(data_n);
}

fn getAverage(vect: &Vec<u128>) -> (f64, f64) {
    let mut average = 0.0;
    let mut sd = 0.0;
    for value in vect {
        average += value.clone() as f64;
    }
    average = average / vect.len() as f64;
    for val in vect {
        sd += (val.clone() as f64 - average)*(val.clone() as f64 - average);
    }
    sd = sd / vect.len() as f64;
    sd = sd.sqrt();
    return (average, sd);
}

// fn generate_data_2000(data_1000: String) -> String {}

fn evaluate_query(simpleQuery: &str, complexQuery: &str, query_type: &str) -> PolarsResult<()> {
    let data_1000 = generate_data_1000().unwrap();
    let db_1000 = &mut SparqlDatabase::new();
    db_1000.parse_turtle(&data_1000.as_str());

    println!("Generated db for 1000 machines");

    let db_2000 = &mut SparqlDatabase::new();
    let data_2000 = generate_data(&data_1000, 2000).unwrap();
    db_2000.parse_turtle(&data_2000.as_str());

    println!("Generated db for 2000 machines");

    let db_3000 = &mut SparqlDatabase::new();
    let data_3000 = generate_data(&data_2000, 3000).unwrap();
    db_3000.parse_turtle(&data_3000.as_str());

    println!("Generated db for 3000 machines");

    let db_4000 = &mut SparqlDatabase::new();
    let data_4000 = generate_data(&data_3000, 4000).unwrap();
    db_4000.parse_turtle(&data_4000.as_str());

    println!("Generated db for 4000 machines");

    let db_5000 = &mut SparqlDatabase::new();
    let data_5000 = generate_data(&data_4000, 5000).unwrap();
    db_5000.parse_turtle(&data_5000.as_str());

    println!("Generated db for 5000 machines");

    let db_6000 = &mut SparqlDatabase::new();
    let data_6000 = generate_data(&data_5000, 6000).unwrap();
    db_6000.parse_turtle(&data_6000.as_str());

    println!("Generated db for 6000 machines");

    let db_7000 = &mut SparqlDatabase::new();
    let data_7000 = generate_data(&data_6000, 7000).unwrap();
    db_7000.parse_turtle(&data_7000.as_str());

    println!("Generated db for 7000 machines");

    let db_8000 = &mut SparqlDatabase::new();
    let data_8000 = generate_data(&data_7000, 8000).unwrap();
    db_8000.parse_turtle(&data_8000.as_str());

    println!("Generated db for 8000 machines");

    let db_9000 = &mut SparqlDatabase::new();
    let data_9000 = generate_data(&data_8000, 9000).unwrap();
    db_9000.parse_turtle(&data_9000.as_str());

    println!("Generated db for 9000 machines");

    let db_10000 = &mut SparqlDatabase::new();
    let data_10000 = generate_data(&data_9000, 10000).unwrap();
    db_10000.parse_turtle(&data_10000.as_str());

    println!("Generated db for 10000 machines");

    let mut avgIntell = Vec::<f64>::new();
    let mut avgDumb = Vec::<f64>::new();
    let mut cards = Vec::<u128>::new();
    let mut sdIntell = Vec::<f64>::new();
    let mut sdDumb = Vec::<f64>::new();

    let (card1000_s, intel1000_s, dumb1000_s) = runQuery(db_1000, simpleQuery, 1000);
    let mut df_1000_s = df![
        "cardinality" => &card1000_s,
        "intelligent runtime" => &intel1000_s,
        "dumb runtime" => &dumb1000_s
    ]?;
    let mut file1000_st = File::create(format!("simple1000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file1000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_1000_s);
    avgIntell.push(getAverage(&intel1000_s).0);
    avgDumb.push(getAverage(&dumb1000_s).0);
    sdIntell.push(getAverage(&intel1000_s).1);
    sdDumb.push(getAverage(&dumb1000_s).1);
    cards.push(1000);

    let (card2000_s, intel2000_s, dumb2000_s) = runQuery(db_2000, simpleQuery, 2000);
    let mut df_2000_s = df![
        "cardinality" => &card2000_s,
        "intelligent runtime" => &intel2000_s,
        "dumb runtime" => &dumb2000_s
    ]?;
    let mut file2000_st = File::create(format!("simple2000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file2000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_2000_s);
    avgIntell.push(getAverage(&intel2000_s).0);
    avgDumb.push(getAverage(&dumb2000_s).0);
    sdIntell.push(getAverage(&intel2000_s).1);
    sdDumb.push(getAverage(&dumb2000_s).1);
    cards.push(2000);

    let (card3000_s, intel3000_s, dumb3000_s) = runQuery(db_3000, simpleQuery, 3000);
    let mut df_3000_s = df![
        "cardinality" => &card3000_s,
        "intelligent runtime" => &intel3000_s,
        "dumb runtime" => &dumb3000_s
    ]?;
    let mut file3000_st = File::create(format!("simple3000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file3000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_3000_s);
    avgIntell.push(getAverage(&intel3000_s).0);
    avgDumb.push(getAverage(&dumb3000_s).0);
    sdIntell.push(getAverage(&intel3000_s).1);
    sdDumb.push(getAverage(&dumb3000_s).1);
    cards.push(3000);

    let (card4000_s, intel4000_s, dumb4000_s) = runQuery(db_4000, simpleQuery, 4000);
    let mut df_4000_s = df![
        "cardinality" => &card4000_s,
        "intelligent runtime" => &intel4000_s,
        "dumb runtime" => &dumb4000_s
    ]?;
    let mut file4000_st = File::create(format!("simple4000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file4000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_4000_s);
    avgIntell.push(getAverage(&intel4000_s).0);
    avgDumb.push(getAverage(&dumb4000_s).0);
    sdIntell.push(getAverage(&intel4000_s).1);
    sdDumb.push(getAverage(&dumb4000_s).1);
    cards.push(4000);

    let (card5000_s, intel5000_s, dumb5000_s) = runQuery(db_5000, simpleQuery, 5000);
    let mut df_5000_s = df![
        "cardinality" => &card5000_s,
        "intelligent runtime" => &intel5000_s,
        "dumb runtime" => &dumb5000_s
    ]?;
    let mut file5000_st = File::create(format!("simple5000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file5000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_5000_s);
    avgIntell.push(getAverage(&intel5000_s).0);
    avgDumb.push(getAverage(&dumb5000_s).0);
    sdIntell.push(getAverage(&intel5000_s).1);
    sdDumb.push(getAverage(&dumb5000_s).1);
    cards.push(5000);

    let (card6000_s, intel6000_s, dumb6000_s) = runQuery(db_6000, simpleQuery, 6000);
    let mut df_6000_s = df![
        "cardinality" => &card6000_s,
        "intelligent runtime" => &intel6000_s,
        "dumb runtime" => &dumb6000_s
    ]?;
    let mut file6000_st = File::create(format!("simple6000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file6000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_6000_s);
    avgIntell.push(getAverage(&intel6000_s).0);
    avgDumb.push(getAverage(&dumb6000_s).0);
    sdIntell.push(getAverage(&intel6000_s).1);
    sdDumb.push(getAverage(&dumb6000_s).1);
    cards.push(6000);

    let (card7000_s, intel7000_s, dumb7000_s) = runQuery(db_7000, simpleQuery, 7000);
    let mut df_7000_s = df![
        "cardinality" => &card7000_s,
        "intelligent runtime" => &intel7000_s,
        "dumb runtime" => &dumb7000_s
    ]?;
    let mut file7000_st = File::create(format!("simple7000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file7000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_7000_s);
    avgIntell.push(getAverage(&intel7000_s).0);
    avgDumb.push(getAverage(&dumb7000_s).0);
    sdIntell.push(getAverage(&intel7000_s).1);
    sdDumb.push(getAverage(&dumb7000_s).1);
    cards.push(7000);

    let (card8000_s, intel8000_s, dumb8000_s) = runQuery(db_8000, simpleQuery, 8000);
    let mut df_8000_s = df![
        "cardinality" => &card8000_s,
        "intelligent runtime" => &intel8000_s,
        "dumb runtime" => &dumb8000_s
    ]?;
    let mut file8000_st = File::create(format!("simple8000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file8000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_8000_s);
    avgIntell.push(getAverage(&intel8000_s).0);
    avgDumb.push(getAverage(&dumb8000_s).0);
    sdIntell.push(getAverage(&intel8000_s).1);
    sdDumb.push(getAverage(&dumb8000_s).1);
    cards.push(8000);

    let (card9000_s, intel9000_s, dumb9000_s) = runQuery(db_9000, simpleQuery, 9000);
    let mut df_9000_s = df![
        "cardinality" => &card9000_s,
        "intelligent runtime" => &intel9000_s,
        "dumb runtime" => &dumb9000_s
    ]?;
    let mut file9000_st = File::create(format!("simple9000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file9000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_9000_s);
    avgIntell.push(getAverage(&intel9000_s).0);
    avgDumb.push(getAverage(&dumb9000_s).0);
    sdIntell.push(getAverage(&intel9000_s).1);
    sdDumb.push(getAverage(&dumb9000_s).1);
    cards.push(9000);

    let (card10000_s, intel10000_s, dumb10000_s) = runQuery(db_10000, simpleQuery, 10000);
    let mut df_10000_s = df![
        "cardinality" => &card10000_s,
        "intelligent runtime" => &intel10000_s,
        "dumb runtime" => &dumb10000_s
    ]?;
    let mut file10000_st = File::create(format!("simple10000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file10000_st)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_10000_s);
    avgIntell.push(getAverage(&intel10000_s).0);
    avgDumb.push(getAverage(&dumb10000_s).0);
    sdIntell.push(getAverage(&intel10000_s).1);
    sdDumb.push(getAverage(&dumb10000_s).1);
    cards.push(10000);

    let mut avgIntellComp = Vec::<f64>::new();
    let mut avgDumbComp = Vec::<f64>::new();
    let mut cardsComp = Vec::<u128>::new();
    let mut sdIntellComp = Vec::<f64>::new();
    let mut sdDumbComp = Vec::<f64>::new();

    let (card1000_c, intel1000_c, dumb1000_c) = runQuery(db_1000, complexQuery, 1000);
    let mut df_1000_c = df![
        "cardinality" => &card1000_c,
        "intelligent runtime" => &intel1000_c,
        "dumb runtime" => &dumb1000_c
    ]?;
    let mut file1000_ct = File::create(format!("complex1000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file1000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_1000_c);
    avgIntellComp.push(getAverage(&intel1000_c).0);
    avgDumbComp.push(getAverage(&dumb1000_c).0);
    sdIntellComp.push(getAverage(&intel1000_c).1);
    sdDumbComp.push(getAverage(&dumb1000_c).1);
    cardsComp.push(1000);

    let (card2000_c, intel2000_c, dumb2000_c) = runQuery(db_2000, complexQuery, 2000);
    let mut df_2000_c = df![
        "cardinality" => &card2000_c,
        "intelligent runtime" => &intel2000_c,
        "dumb runtime" => &dumb2000_c
    ]?;
    let mut file2000_ct = File::create(format!("complex2000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file2000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_2000_c);
    avgIntellComp.push(getAverage(&intel2000_c).0);
    avgDumbComp.push(getAverage(&dumb2000_c).0);
    sdIntellComp.push(getAverage(&intel2000_c).1);
    sdDumbComp.push(getAverage(&dumb2000_c).1);
    cardsComp.push(2000);

    let (card3000_c, intel3000_c, dumb3000_c) = runQuery(db_3000, complexQuery, 3000);
    let mut df_3000_c = df![
        "cardinality" => &card3000_c,
        "intelligent runtime" => &intel3000_c,
        "dumb runtime" => &dumb3000_c
    ]?;
    let mut file3000_ct = File::create(format!("complex3000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file3000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_3000_c);
    avgIntellComp.push(getAverage(&intel3000_c).0);
    avgDumbComp.push(getAverage(&dumb3000_c).0);
    sdIntellComp.push(getAverage(&intel3000_c).1);
    sdDumbComp.push(getAverage(&dumb3000_c).1);
    cardsComp.push(3000);

    let (card4000_c, intel4000_c, dumb4000_c) = runQuery(db_4000, complexQuery, 4000);
    let mut df_4000_c = df![
        "cardinality" => &card4000_c,
        "intelligent runtime" => &intel4000_c,
        "dumb runtime" => &dumb4000_c
    ]?;
    let mut file4000_ct = File::create(format!("complex4000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file4000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_4000_c);
    avgIntellComp.push(getAverage(&intel4000_c).0);
    avgDumbComp.push(getAverage(&dumb4000_c).0);
    sdIntellComp.push(getAverage(&intel4000_c).1);
    sdDumbComp.push(getAverage(&dumb4000_c).1);
    cardsComp.push(4000);

    let (card5000_c, intel5000_c, dumb5000_c) = runQuery(db_5000, complexQuery, 5000);
    let mut df_5000_c = df![
        "cardinality" => &card5000_c,
        "intelligent runtime" => &intel5000_c,
        "dumb runtime" => &dumb5000_c
    ]?;
    let mut file5000_ct = File::create(format!("complex5000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file5000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_5000_c);
    avgIntellComp.push(getAverage(&intel5000_c).0);
    avgDumbComp.push(getAverage(&dumb5000_c).0);
    sdIntellComp.push(getAverage(&intel5000_c).1);
    sdDumbComp.push(getAverage(&dumb5000_c).1);
    cardsComp.push(5000);

    let (card6000_c, intel6000_c, dumb6000_c) = runQuery(db_6000, complexQuery, 6000);
    let mut df_6000_c = df![
        "cardinality" => &card6000_c,
        "intelligent runtime" => &intel6000_c,
        "dumb runtime" => &dumb6000_c
    ]?;
    let mut file6000_ct = File::create(format!("complex6000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file6000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_6000_c);
    avgIntellComp.push(getAverage(&intel6000_c).0);
    avgDumbComp.push(getAverage(&dumb6000_c).0);
    sdIntellComp.push(getAverage(&intel6000_c).1);
    sdDumbComp.push(getAverage(&dumb6000_c).1);
    cardsComp.push(6000);

    let (card7000_c, intel7000_c, dumb7000_c) = runQuery(db_7000, complexQuery, 7000);
    let mut df_7000_c = df![
        "cardinality" => &card7000_c,
        "intelligent runtime" => &intel7000_c,
        "dumb runtime" => &dumb7000_c
    ]?;
    let mut file7000_ct = File::create(format!("complex7000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file7000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_7000_c);
    avgIntellComp.push(getAverage(&intel7000_c).0);
    avgDumbComp.push(getAverage(&dumb7000_c).0);
    sdIntellComp.push(getAverage(&intel7000_c).1);
    sdDumbComp.push(getAverage(&dumb7000_c).1);
    cardsComp.push(7000);

    let (card8000_c, intel8000_c, dumb8000_c) = runQuery(db_8000, complexQuery, 5000);
    let mut df_8000_c = df![
        "cardinality" => &card8000_c,
        "intelligent runtime" => &intel8000_c,
        "dumb runtime" => &dumb8000_c
    ]?;
    let mut file8000_ct = File::create(format!("complex8000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file8000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_8000_c);
    avgIntellComp.push(getAverage(&intel8000_c).0);
    avgDumbComp.push(getAverage(&dumb8000_c).0);
    sdIntellComp.push(getAverage(&intel8000_c).1);
    sdDumbComp.push(getAverage(&dumb8000_c).1);
    cardsComp.push(8000);

    let (card9000_c, intel9000_c, dumb9000_c) = runQuery(db_9000, complexQuery, 9000);
    let mut df_9000_c = df![
        "cardinality" => &card9000_c,
        "intelligent runtime" => &intel9000_c,
        "dumb runtime" => &dumb9000_c
    ]?;
    let mut file9000_ct = File::create(format!("complex9000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file9000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_9000_c);
    avgIntellComp.push(getAverage(&intel9000_c).0);
    avgDumbComp.push(getAverage(&dumb9000_c).0);
    sdIntellComp.push(getAverage(&intel9000_c).1);
    sdDumbComp.push(getAverage(&dumb9000_c).1);
    cardsComp.push(9000);

    let (card10000_c, intel10000_c, dumb10000_c) = runQuery(db_10000, complexQuery, 10000);
    let mut df_10000_c = df![
        "cardinality" => &card10000_c,
        "intelligent runtime" => &intel10000_c,
        "dumb runtime" => &dumb10000_c
    ]?;
    let mut file10000_ct = File::create(format!("complex10000{}.csv", query_type)).expect("could not create file");
    CsvWriter::new(&mut file10000_ct)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_10000_c);
    avgIntellComp.push(getAverage(&intel10000_c).0);
    avgDumbComp.push(getAverage(&dumb10000_c).0);
    sdIntellComp.push(getAverage(&intel10000_c).1);
    sdDumbComp.push(getAverage(&dumb10000_c).1);
    cardsComp.push(10000);

    let mut plotDf1 = df![
        "cardinality" => cards,
        "average intelligent" => avgIntell,
        "average dumb" => avgDumb,
        "standard deviation intelligent" => sdIntell,
        "standard deviation dumb" => sdDumb
    ]?;
    let mut plotfile1 = File::create(format!("{}PlottingResultsSimple.csv", query_type)).expect("could not create file");
    
    CsvWriter::new(&mut plotfile1)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut plotDf1);

    let mut plotDf = df![
        "cardinality" => cardsComp,
        "average intelligent" => avgIntellComp,
        "average dumb" => avgDumbComp,
        "standard deviation intelligent" => sdIntellComp,
        "standard deviation dumb" => sdDumbComp
    ]?;
    let mut plotfile = File::create(format!("{}PlottingResultsComplex.csv", query_type)).expect("could not create file");
    
    CsvWriter::new(&mut plotfile)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut plotDf);

    return Ok(());
}

fn main() -> PolarsResult<()> {
    let simpleQuery1 = r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX ext: <http://www.thesisextension.org/runml#>
    SELECT ?machine ?prediction ?runtime ?job ?salary
WHERE {
	?machine <http://example.org#humidValue> ?humid.
	?machine <http://example.org#tempValue> ?temp.
    ?machine <http://example.org#powerValue> ?energy.
    ?machine <http://example.org#vibrValue> ?vibr.
	?job <http://example.org#hasAccess> ?machine.
	?job <http://example.org#hasSalary> ?salary.
	RUN {?humid, ?temp, ?energy, ?vibr} ON ?runtime TO ?prediction.
}"#;

    let complexQuery1 = r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX ext: <http://www.thesisextension.org/runml#>
    SELECT ?person ?machine ?runtime ?prediction ?job ?animal
WHERE {
	?machine <http://example.org#humidValue> ?humid.
	?machine <http://example.org#tempValue> ?temp.
    ?machine <http://example.org#powerValue> ?energy.
    ?machine <http://example.org#vibrValue> ?vibr.
	?job <http://example.org#hasAccess> ?machine.
	?person <http://example.org#works> ?job.
	?person <http://example.org#hasPet> ?animal.
	RUN {?humid, ?temp, ?energy, ?vibr} ON ?runtime TO ?prediction.
}"#;
    
    evaluate_query(simpleQuery1, complexQuery1, "tree");

    let simpleQuery2 = r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX ext: <http://www.thesisextension.org/runml#>
    SELECT ?machine ?prediction ?runtime ?alarm ?alarmAction
WHERE {
	?machine <http://example.org#humidValue> ?humid.
	?machine <http://example.org#tempValue> ?temp.
    ?machine <http://example.org#powerValue> ?energy.
    ?machine <http://example.org#vibrValue> ?vibr.
	?machine <http://example.org#hasAlarm> ?alarm.
	RUN {?humid, ?temp, ?energy, ?vibr} ON ?runtime TO ?prediction .
	?alarm <http://example.org#activationType> ?alarmAction .
}"#;

// ?prediction <http://example.org#activates> ?alarm.

    let complexQuery2 = r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX ext: <http://www.thesisextension.org/runml#>
    SELECT ?machine ?prediction ?runtime ?alarm ?alarmAction ?impactLevel 
WHERE {
	?machine <http://example.org#humidValue> ?humid.
	?machine <http://example.org#tempValue> ?temp.
    ?machine <http://example.org#powerValue> ?energy.
    ?machine <http://example.org#vibrValue> ?vibr.
	?machine <http://example.org#hasAlarm> ?alarm.
	RUN {?humid, ?temp, ?energy, ?vibr} ON ?runtime TO ?prediction.
	?alarm <http://example.org#activationType> ?alarmAction.
    ?alarmAction <http://example.org#informs> ?team.
}"#;
    evaluate_query(simpleQuery2, complexQuery2, "linearStar");

    let simpleQuery3 = r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX ext: <http://www.thesisextension.org/runml#>
    SELECT ?machine ?runtime ?prediction ?alarm ?alarmAction
WHERE {
	?machine <http://example.org#humidValue> ?humid.
	?machine <http://example.org#tempValue> ?temp.
    ?machine <http://example.org#powerValue> ?energy.
    ?machine <http://example.org#vibrValue> ?vibr.
	?machine <http://example.org#hasAlarm> ?alarm.
	RUN {?humid, ?temp, ?energy, ?vibr} ON ?runtime TO ?prediction.
	?alarm <http://example.org#activationType> ?alarmAction.
	?alarmAction <http://example.org#supportedBy> ?machine.
}"#;

    let complexQuery3 = r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX ext: <http://www.thesisextension.org/runml#>
    SELECT ?machine ?runtime ?prediction ?alarm ?alarmAction ?team
WHERE {
	?machine <http://example.org#humidValue> ?humid.
	?machine <http://example.org#tempValue> ?temp.
    ?machine <http://example.org#powerValue> ?energy.
    ?machine <http://example.org#vibrValue> ?vibr.
	?machine <http://example.org#hasAlarm> ?alarm.
	RUN {?humid, ?temp, ?energy, ?vibr} ON ?runtime TO ?prediction.
	?alarm <http://example.org#activationType> ?alarmAction.
	?alarmAction <http://example.org#informs> ?team.
    ?team <http://example.org#maintains> ?machine.
}"#;

    evaluate_query(simpleQuery3, complexQuery3, "cycleStar");

    // let simpleQuery2 = 
    return Ok(());
}

fn runQuery(db: &mut SparqlDatabase, simpleQuery: &str, machines: u128) -> (Vec<u128>, Vec<u128>, Vec<u128>) {
    let optimizer = &mut Streamertail::new(&db);

    let mut intelligentRuns = Vec::new();
    let mut naiveRuns = Vec::new();
    let mut cardinality = Vec::new();

    let output = parse_sparql_query(simpleQuery);
    let (
        _,
        (
            insert_clause,
            mut variables,
            patterns,
            filters,
            group_vars,
            mut parsed_prefixes,
            values_clause,
            binds,
            subqueries,
            limit,
            _,
            order_conditions,
            ml_run_clause
        ),
        ) = output.unwrap();

    let mut selected_variables: Vec<(String, String)> = Vec::new();
    let mut aggregation_vars: Vec<(&str, &str, &str)> = Vec::new();
    process_variables(&mut selected_variables, &mut aggregation_vars, variables);

    let naive_lo = dumb_build_logical_plan(
        selected_variables
                    .iter()
                    .map(|(t, v)| (t.as_str(), v.as_str()))
                    .collect(), 
        patterns.clone(), 
        filters.clone(), 
        &parsed_prefixes, 
        db, 
        &binds, 
        values_clause.as_ref(), 
        ml_run_clause.clone()
    );

    let intell_lo = build_logical_plan(
        selected_variables
                .iter()
                .map(|(t, v)| (t.as_str(), v.as_str()))
                .collect(), 
        patterns, 
        filters, 
        &parsed_prefixes, 
        db, 
        &binds, 
        values_clause.as_ref(), 
        ml_run_clause.clone()
    );

    let naive_po = optimizer.find_best_plan(&naive_lo, db);
    let intell_po = optimizer.find_best_plan(&intell_lo, db);

    let start1 = Instant::now();
    let naive_result = naive_po.execute_with_ids(db);
    let naive_time = start1.elapsed().as_millis();
    println!("Naive Time = {:?}", naive_time);

    let start2 = Instant::now();
    let intell_result = intell_po.execute_with_ids(db);
    let intell_time = start2.elapsed().as_millis();
    println!("Intelligent Time = {:?}", intell_time);

    for i in 1..30{
        let start1 = Instant::now();
        let naive_result = naive_po.execute_with_ids(db);
        let naive_time = start1.elapsed().as_millis();
        naiveRuns.push(naive_time);
        println!("Naive Time = {:?}", naive_time);

        let start2 = Instant::now();
        let intell_result = intell_po.execute_with_ids(db);
        let intell_time = start2.elapsed().as_millis();
        intelligentRuns.push(intell_time);
        println!("Intelligent Time = {:?}", intell_time);

        cardinality.push(machines);
    }

    return (cardinality, intelligentRuns, naiveRuns);
}
    