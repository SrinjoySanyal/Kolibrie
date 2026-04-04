#[cfg(test)]
mod tests {
    use kolibrie::sparql_database::SparqlDatabase;
    use kolibrie::streamertail_optimizer::operators::{LogicalOperator, PhysicalOperator};
    use kolibrie::parser::parse_sparql_query;
    use kolibrie::streamertail_optimizer::utils::*;
    use kolibrie::execute_query::process_variables;
    use kolibrie::streamertail_optimizer::optimizer::Streamertail;
    #[test]
    fn ml_integration_test() {
        let db = &mut SparqlDatabase::new();            

        // Employee dataset in Turtle format
        let turtle_data = r#"
            <http://example.org#room101> rdf:type <http://example.org#Room> .
            <http://example.org#room101> sensor:temperature "22.5" .
            <http://example.org#room101> sensor:humidity "45.0" .
            <http://example.org#room101> sensor:occupancy "5" .
            
            <http://example.org#room102> rdf:type <http://example.org#Room> .
            <http://example.org#room102> sensor:temperature "23.0" .
            <http://example.org#room102> sensor:humidity "52.0" .
            <http://example.org#room102> sensor:occupancy "8" .
            
            <http://example.org#room103> rdf:type <http://example.org#Room> .
            <http://example.org#room103> sensor:temperature "27.2" .
            <http://example.org#room103> sensor:humidity "48.0" .
            <http://example.org#room103> sensor:occupancy "3" .

            <http://example.org#roomML> rdf:type mls:Run .
            <http://example.org#roomML> mls:hasOutput rf_temperature_predictor .
            rf_temperature_predictor rdf:type mls:Model . 

            <http://example.org#roomML2> rdf:type mls:Run .
            <http://example.org#roomML2> mls:hasOutput gb_temperature_predictor .
            gb_temperature_predictor rdf:type mls:Model .

            <http://example.org#roomML> sensor:connectsTo sensor:alarm.
        "#;
        db.parse_turtle(turtle_data);
        let optimizer = &mut Streamertail::new(&db);

        let sparql = r#"
        SELECT ?building ?prediction WHERE {
            ?building sensor:temperature ?temp.
            ?building sensor:humidity ?humidity.
            RUN {?temp, ?humidity, ?occupancy} ON ?r TO ?prediction.
            ?building sensor:occupancy ?occupancy.
        }
        "#;

        let output = parse_sparql_query(sparql);
        // print!("{}", output);
        assert!(output.is_ok());

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

        assert!(ml_run_clause.is_some());
        // println!("ml run clause = {ml_run_clause:?}");

        let mut selected_variables: Vec<(String, String)> = Vec::new();
        let mut aggregation_vars: Vec<(&str, &str, &str)> = Vec::new();
        process_variables(&mut selected_variables, &mut aggregation_vars, variables);

        let produced_logical_operator = build_logical_plan(
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

        // println!("logical operator = {produced_logical_operator:?}");
        
        let produced_physical_operator = optimizer.find_best_plan(&produced_logical_operator);
        // println!("best plan = {produced_physical_operator:?}");
        let result = produced_physical_operator.execute(db);
        println!("final results = {result:?}");
        assert_eq!(result.len(), 9);
    }
}