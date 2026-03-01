/*
 * Copyright © 2024 Volodymyr Kadzhaia
 * Copyright © 2024 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use super::operators::{LogicalOperator, PhysicalOperator};
use super::types::Condition;
// use crate::parser::predicate;
use crate::parser::*;
use crate::sparql_database::SparqlDatabase;
use nom::combinator::not;
use shared::query::{FilterExpression, MLClause, SubQuery, ValuesClause};
use shared::terms::{Term, TriplePattern};
use std::collections::HashMap;
use crate::execute_query;
use std::sync::Arc;

/// Extracts a triple pattern from a physical operator if it's a scan operation
pub fn extract_pattern(op: &PhysicalOperator) -> Option<&TriplePattern> {
    match op {
        PhysicalOperator::TableScan { pattern } | PhysicalOperator::IndexScan { pattern } => {
            Some(pattern)
        }
        // If it's a Filter, keep searching in its child
        PhysicalOperator::Filter { input, .. } => extract_pattern(input),
        _ => None,
    }
}

/// Checks if a pattern contains a specific variable
pub fn pattern_contains_variable(pattern: &TriplePattern, var: &str) -> bool {
    matches!(&pattern.0, Term::Variable(v) if v == var)
        || matches!(&pattern.1, Term::Variable(v) if v == var)
        || matches!(&pattern.2, Term::Variable(v) if v == var)
}

/// Estimates the selectivity of an operator for optimization purposes
pub fn estimate_operator_selectivity(op: &LogicalOperator, _database: &SparqlDatabase) -> u64 {
    match op {
        LogicalOperator::Scan { pattern } => {
            let bound_count = count_bound_terms(pattern);

            match bound_count {
                3 => 1, // Highest priority - fully bound
                2 => 2, // High priority - two bounds
                1 => 3, // Medium priority - one bound
                0 => 4, // Lowest priority - no bounds
                _ => 5,
            }
        }
        LogicalOperator::Selection { predicate, .. } => {
            // Selections are generally high priority due to filtering
            estimate_operator_selectivity(predicate, _database) + 10
        }
        LogicalOperator::Join { left, right } => {
            // Join cost depends on both sides
            let left_sel = estimate_operator_selectivity(left, _database);
            let right_sel = estimate_operator_selectivity(right, _database);
            left_sel + right_sel + 5
        }
        LogicalOperator::Projection { predicate, .. } => {
            // Projection doesn't change selectivity much
            estimate_operator_selectivity(predicate, _database) + 1
        }
        LogicalOperator::Buffer { .. } => {10000}
        LogicalOperator::Subquery { inner, .. } => {
            estimate_operator_selectivity(inner, _database) + 15
        }
        LogicalOperator::Bind { input, .. } => {
            estimate_operator_selectivity(input, _database) + 2
        }
        LogicalOperator::Values { values, .. } => {
            values.len() as u64
        }
        LogicalOperator::MLPredict { input, input_variables, .. } => {
            let base_selectivity = estimate_operator_selectivity(input, _database);
            let ml_overhead = 50 + (input_variables.len() as u64 * 10);
            base_selectivity + ml_overhead
        }
    }
}

/// Counts the number of bound terms (constants) in a triple pattern
fn count_bound_terms(pattern: &TriplePattern) -> usize {
    let mut count = 0;

    if matches!(&pattern.0, Term::Constant(_)) {
        count += 1;
    }
    if matches!(&pattern.1, Term::Constant(_)) {
        count += 1;
    }
    if matches!(&pattern.2, Term::Constant(_)) {
        count += 1;
    }

    count
}

/// Builds an optimized logical plan from query components
pub fn build_logical_plan(
    variables: Vec<(&str, &str)>,
    patterns: Vec<(&str, &str, &str)>,
    filters: Vec<FilterExpression>,
    prefixes: &HashMap<String, String>,
    database: &mut SparqlDatabase,
    binds: &[(&str, Vec<&str>, &str)],
    values_clause: Option<&ValuesClause>,
    ml_run_clause: Option<MLClause>
) -> LogicalOperator {
    // Create base operator from VALUES if present, otherwise empty join base
    let mut result = if let Some(values_clause) = values_clause {
        // Convert ValuesClause to LogicalOperator::Values
        let variables: Vec<String> = values_clause
            .variables
            .iter()
            .map(|v| v.to_string())
            .collect();

        let values: Vec<Vec<Option<String>>> = values_clause
            .values
            .iter()
            .map(|row| {
                row.iter()
                .map(|value| match value {
                    shared::query::Value::Term(term) => Some(term.clone()),
                    shared::query::Value::Undef => None,
                })
                .collect()
            })
            .collect();

        LogicalOperator::values(variables, values)
    } else {
        // Start with first pattern as before
        let first_pattern = if patterns.is_empty() {
            // Empty query - return a minimal scan
            let pattern = (
                Term::Variable("?s".to_string()),
                Term::Variable("?p".to_string()),
                Term::Variable("?o".to_string()),
            );
            LogicalOperator::scan(pattern)
        } else {
            let (subject_str, predicate_str, object_str) = patterns[0];
            let pattern = convert_pattern_to_triple(
                subject_str,
                predicate_str,
                object_str,
                prefixes,
                database
            );
            LogicalOperator::scan(pattern)
        };
        first_pattern
    };

    // If we have VALUES, join it with all patterns
    // Otherwise, join patterns together as before
    let start_idx = if values_clause.is_some() { 0 } else { 1 };

    for (subject_str, predicate_str, object_str) in patterns.iter().skip(start_idx) {
        let pattern = convert_pattern_to_triple(
            subject_str,
            predicate_str,
            object_str,
            prefixes,
            database,
        );
        let scan_op = LogicalOperator::scan(pattern);
        result = LogicalOperator::join(result, scan_op);
    }

    // Apply filters that couldn't be pushed down
    for filter in filters {
        let condition = convert_filter_to_condition(&filter);
        result = LogicalOperator::selection(result, condition);
    }

    // Apply BIND clauses
    for (func_name, args, output_var) in binds {
        let function_name = func_name.to_string();
        let arguments: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let output_variable = output_var.to_string();

        result = LogicalOperator::bind(result, function_name, arguments, output_variable);
    }

    // Apply projection if specific variables were requested
    if !variables.is_empty() {
        let var_names: Vec<String> = variables.into_iter().map(|(_, v)| v.to_string()).collect();
        result = LogicalOperator::projection(result, var_names);
    }
    

    if let Some(ml_run_clause_val) = ml_run_clause {
        let mut ml_run_clause_value = ml_run_clause_val.clone();
        // variable storing the name of the SPARQL variable representing ML models themselves
        let ml_models_var = get_ml_var_name(result.clone(), &ml_run_clause_value.on.to_string(), database);
        let mut ml_retrieved_in_query = false;
        let mut maxdepth = 0; 
        // find the least nested join containing all the variables passed as argument to RUN
        for var in &ml_run_clause_value.run {
            // least nested operator has a higher depth than a more nested operator
            let mut initial = result.clone();
            let mut depth = patterns.len();
            let var_str = var.to_string();
            loop {
                match initial {
                    LogicalOperator::Scan { pattern } => {
                        if let Term::Variable(ref string) = pattern.0 {
                            if *string == ml_run_clause_value.on.to_string() {
                                ml_retrieved_in_query = true;
                            }
                            if *string == var_str{
                                if depth > maxdepth {
                                    maxdepth = depth;
                                }
                                break;
                            }
                        }
                        if let Term::Variable(ref string) = pattern.1 {
                            if *string == ml_run_clause_value.on.to_string() {
                                ml_retrieved_in_query = true;
                            }
                            if *string == var_str{
                                if (depth > maxdepth) {
                                    maxdepth = depth;
                                }
                                break;
                            }
                        }
                        if let Term::Variable(ref string) = pattern.2 {
                            if *string == ml_run_clause_value.on.to_string() {
                                ml_retrieved_in_query = true;
                            }
                            if *string == var_str{
                                if (depth > maxdepth) {
                                    maxdepth = depth;
                                }
                                break;
                            }
                        }
                        break;
                    },
                    LogicalOperator::Values { variables, values } => {break;},
                    LogicalOperator::Projection { ref predicate, variables } => {
                        initial = *predicate.clone();
                    }
                    LogicalOperator::Bind { ref input, function_name, arguments, output_variable } => {
                        initial = *input.clone();
                    }
                    LogicalOperator::Selection { ref predicate, condition } => {
                        initial = *predicate.clone();
                    }
                    LogicalOperator::Join { ref left, ref right } => {
                        match right.as_ref() {
                            LogicalOperator::Scan { ref pattern } => {
                                if let Term::Variable(ref string) = pattern.0 {
                                    if *string == ml_run_clause_value.on.to_string() {
                                        ml_retrieved_in_query = true;
                                    }
                                    if *string == var_str{
                                        if (depth > maxdepth) {
                                            maxdepth = depth;
                                        }
                                        break;
                                    }
                                }
                                if let Term::Variable(ref string) = pattern.1 {
                                    if *string == ml_run_clause_value.on.to_string() {
                                        ml_retrieved_in_query = true;
                                    }
                                    if *string == var_str{
                                        if (depth > maxdepth) {
                                            maxdepth = depth;
                                        }
                                        break;
                                    }
                                }
                                if let Term::Variable(ref string) = pattern.2 {
                                    if *string == ml_run_clause_value.on.to_string() {
                                        ml_retrieved_in_query = true;
                                    }
                                    if *string == var_str{
                                        if (depth > maxdepth) {
                                            maxdepth = depth;
                                        }
                                        break;
                                    }
                                }
                                initial = (**left).clone();
                                depth -= 1;
                            },
                            _ => {
                                initial = (**left).clone();
                                depth -= 1;
                            }
                        }
                    }
                    _ => {break;}
                }
            }
        }
        let mut insertionMLRun = result.clone();
        let mut currentDepth = patterns.len();
        // the left argument of the LogicalOperator::Join at this replacementDepth level 
        // will have as its left argument a LogicalOperator that collects all the variables passed as argument to
        // the RUN clause of the Run Ml Clause 
        let replacementDepth = maxdepth + 1;
        
        if ml_models_var.is_none() {
            if ml_retrieved_in_query {

                // get the operator to retrieve the ml models on which to run the code, without the last projection layer
                let mut ml_retrieval_logical_op = get_least_nested_join_with_scan_on_str(ml_run_clause_val.on.to_string(), result.clone(), database).unwrap();
                let pattern2 = convert_pattern_to_triple(ml_run_clause_value.on, "mls:hasOutput", "?mlmodel", prefixes, database);
                let pattern3 = convert_pattern_to_triple("?mlmodel", "rdf:type", "mls:Model", prefixes, database);

                ml_retrieval_logical_op = LogicalOperator::join (ml_retrieval_logical_op, LogicalOperator::scan(pattern2));
                ml_retrieval_logical_op = LogicalOperator::join(ml_retrieval_logical_op, LogicalOperator::scan(pattern3));
                ml_retrieval_logical_op = LogicalOperator::projection(ml_retrieval_logical_op, Vec::from(["?mlmodel".to_string()]));

                result = insert_ml_run_clause_logical_op(currentDepth, &replacementDepth, &insertionMLRun, Some(ml_retrieval_logical_op), &(ml_run_clause_value.run), ml_run_clause_value.to);
            } 
            else {
                let pattern1 = convert_pattern_to_triple(ml_run_clause_value.on, "rdf:type", "mls:Run", prefixes, database);
                let pattern2 = convert_pattern_to_triple(ml_run_clause_value.on, "mls:hasOutput", "?mlmodel", prefixes, database);
                let pattern3 = convert_pattern_to_triple("?mlmodel", "rdf:type", "mls:Model", prefixes, database);

                let mut ml_retrieval_logical_op = LogicalOperator::scan(pattern1);
                ml_retrieval_logical_op = LogicalOperator::join (ml_retrieval_logical_op, LogicalOperator::scan(pattern2));
                ml_retrieval_logical_op = LogicalOperator::join(ml_retrieval_logical_op, LogicalOperator::scan(pattern3));
                ml_retrieval_logical_op = LogicalOperator::projection(ml_retrieval_logical_op, Vec::from(["?mlmodel".to_string()]));

                
                result = insert_ml_run_clause_logical_op(currentDepth, &replacementDepth, &insertionMLRun, Some(ml_retrieval_logical_op), &(ml_run_clause_value.run), ml_run_clause_value.to);
            }
        }
        if let Some(join_logical_operator_with_ml_var) = ml_models_var {
            let mut ml_retrieval_logical_op = get_least_nested_join_with_scan_on_str(join_logical_operator_with_ml_var.clone(), result.clone(), database).unwrap();
            // let retndop = format!("{ml_retrieval_logical_op:?}");
            // println!("returned logical op {retndop}");
            ml_retrieval_logical_op = LogicalOperator::projection(ml_retrieval_logical_op, Vec::from([join_logical_operator_with_ml_var]));
            let retndop2 = format!("{ml_retrieval_logical_op:?}");

            result = insert_ml_run_clause_logical_op(currentDepth, &replacementDepth, &insertionMLRun, Some(ml_retrieval_logical_op), &(ml_run_clause_value.run), ml_run_clause_value.to);
        }
        
    }

    result
}

fn get_least_nested_join_with_scan_on_str(
    targetStr: String,
    logicalOp: LogicalOperator,
    database: &mut SparqlDatabase
) -> Option<LogicalOperator> {
        match logicalOp {
            LogicalOperator::Scan { pattern } => {
                if let Term::Variable(ref string) = pattern.0 {
                    if *string == targetStr {
                        return Some(LogicalOperator::scan(pattern.clone()));
                    }
                }
                if let Term::Variable(ref string) = pattern.1 {
                    if *string == targetStr {
                        return Some(LogicalOperator::scan(pattern.clone()));
                    }
                }
                if let Term::Variable(ref string) = pattern.2 {
                    if *string == targetStr {
                        return Some(LogicalOperator::scan(pattern.clone()));
                    }
                }
                
                let dict_extracted = database.dictionary.write().unwrap();
                if let Term::Constant(ref id) = pattern.0 {
                    let cst_for_mlmodel = dict_extracted.decode(*id);
                    if cst_for_mlmodel.unwrap() == targetStr {
                        return Some(LogicalOperator::scan(pattern.clone()));
                    }
                }
                if let Term::Constant(ref id) = pattern.1 {
                    let cst_for_mlmodel = dict_extracted.decode(*id);
                    if cst_for_mlmodel.unwrap() == targetStr {
                        return Some(LogicalOperator::scan(pattern.clone()));
                    }
                }
                if let Term::Constant(ref id) = pattern.2 {
                    let cst_for_mlmodel = dict_extracted.decode(*id);
                    if cst_for_mlmodel.unwrap() == targetStr {
                        return Some(LogicalOperator::scan(pattern.clone()));
                    }
                }
                return None;
            },
            // LogicalOperator::Values { variables, values } => {return None;},
            LogicalOperator::Projection { ref predicate, variables } => {
                return get_least_nested_join_with_scan_on_str(targetStr, (**predicate).clone(), database);
            }
            LogicalOperator::Bind { ref input, function_name, arguments, output_variable } => {
                return get_least_nested_join_with_scan_on_str(targetStr, (**input).clone(), database)
            }
            LogicalOperator::Selection { ref predicate, condition } => {
                return get_least_nested_join_with_scan_on_str(targetStr, (**predicate).clone(), database);
            }
            LogicalOperator::Join { ref left, ref right } => {
                match right.as_ref() {
                    LogicalOperator::Scan { pattern } => {
                        if let Term::Variable(ref string) = pattern.0 {
                            if *string == targetStr {
                                return Some(LogicalOperator::join(*left.clone(), *right.clone()));
                            }
                        }
                        if let Term::Variable(ref string) = pattern.1 {
                            if *string == targetStr {
                                return Some(LogicalOperator::join(*left.clone(), *right.clone()));
                            }
                        }
                        if let Term::Variable(ref string) = pattern.2 {
                            if *string == targetStr {
                                return Some(LogicalOperator::join(*left.clone(), *right.clone()));
                            }
                        }
                        
                        if let Term::Constant(ref id) = pattern.0 {
                            let cst_for_mlmodel = &database.dictionary.write().unwrap().decode(*id).unwrap().to_string();
                            if *cst_for_mlmodel == targetStr {
                                return Some(LogicalOperator::join(*left.clone(), *right.clone()));
                            }
                        }
                        if let Term::Constant(ref id) = pattern.1 {
                            let cst_for_mlmodel = &database.dictionary.write().unwrap().decode(*id).unwrap().to_string();
                            if *cst_for_mlmodel == targetStr {
                                return Some(LogicalOperator::join(*left.clone(), *right.clone()));
                            }
                        }
                        if let Term::Constant(ref id) = pattern.2 {
                            let cst_for_mlmodel = &database.dictionary.write().unwrap().decode(*id).unwrap().to_string();;
                            if *cst_for_mlmodel == targetStr {
                                return Some(LogicalOperator::join(*left.clone(), *right.clone()));
                            }
                        }
                        database.dictionary.clear_poison();
                        return get_least_nested_join_with_scan_on_str(targetStr, (**left).clone(), database);
                    },
                    _ => {
                        return get_least_nested_join_with_scan_on_str(targetStr, (**left).clone(), database);
                    }
                }
            }
            _ => {
                println!("here I am");
                return None;
            }
    }
}

fn get_ml_var_name(
    mut operator: LogicalOperator,
    runtimeVar: &String,
    database: &mut SparqlDatabase
) -> Option<String> {
    match operator {
        LogicalOperator::Projection { predicate, variables } => {
            return get_ml_var_name(*predicate, runtimeVar, database);
        }
        LogicalOperator::Selection { predicate, condition } => {
            return get_ml_var_name(*predicate, runtimeVar, database);
        }
        LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
            return get_ml_var_name(*input, runtimeVar, database);
        }
        LogicalOperator::Scan { pattern } => {
            if let Term::Variable(var) = pattern.0 {
                if var == *runtimeVar {
                    if let Term::Constant(id) = pattern.1 {
                        if let Some(pred_var) = database.dictionary.write().unwrap().decode(id) {
                            if pred_var.to_string() == resolve_with_prefixes("mls:hasOutput", &database.prefixes) {
                                if let Term::Variable(ml_var) = pattern.2 {
                                    return Some(ml_var)
                                }
                            }
                        }
                    }
                }
            }
            return None
        }
        LogicalOperator::Join { left, right } => {
            match right.as_ref() {
                LogicalOperator::Scan { pattern } => {
                    if let Term::Variable(var) = pattern.0.clone() {
                        if var == *runtimeVar {
                            if let Term::Constant(id) = pattern.1 {
                                if let Some(pred_var) = database.dictionary.write().unwrap().decode(id) {
                                    if pred_var.to_string() == resolve_with_prefixes("mls:hasOutput", &database.prefixes) {
                                        if let Term::Variable(ml_var) = pattern.2.clone() {
                                            return Some(ml_var)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return get_ml_var_name(*left, runtimeVar, database)
                }
                _ => {return get_ml_var_name(*left, runtimeVar, database)}
            }
        }
        _ => {return None}
    }
}

fn insert_ml_run_clause_logical_op(
    currentDepth: usize,
    replacementDepth: &usize,
    logicalOp: &LogicalOperator,
    ml_model_retrieval: Option<LogicalOperator>,
    input_vars: &Vec<&str>,
    output_var: &str
) -> LogicalOperator {
    if currentDepth == *replacementDepth {
        match logicalOp {
            LogicalOperator::Join { left, right } => {
                if let Some(ml_model_retrieval_operator) = ml_model_retrieval {
                    let mlop = LogicalOperator::run_ml_clause_lo(left.as_ref().clone(), ml_model_retrieval_operator, input_vars.clone().iter().map(|inp| inp.to_string()).collect(), output_var.to_string());
                    return LogicalOperator::join(mlop, *right.clone());
                }
            }
            LogicalOperator::Projection { predicate, variables } => {
            let pred_new = insert_ml_run_clause_logical_op(currentDepth, replacementDepth, predicate, ml_model_retrieval, input_vars, output_var);
                return LogicalOperator::Projection { 
                    predicate: Box::new(pred_new), 
                    variables: variables.clone() 
                }
            }
            LogicalOperator::Selection { predicate, condition } => {
                let pred_new = insert_ml_run_clause_logical_op(currentDepth, replacementDepth, predicate, ml_model_retrieval, input_vars, output_var);
                return LogicalOperator::Selection { 
                    predicate: Box::new(pred_new), 
                    condition: condition.clone() 
                }
            }
            LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
                let input_new = insert_ml_run_clause_logical_op(currentDepth, replacementDepth, input, ml_model_retrieval, input_vars, output_var);
                return LogicalOperator::Bind { 
                    input: Box::new(input_new), 
                    function_name: function_name.clone(), 
                    arguments: arguments.clone(), 
                    output_variable: output_variable.clone() 
                }
                
            }
            _ => {return logicalOp.clone()}
        }
    }
    match logicalOp {
        LogicalOperator::Projection { predicate, variables } => {
            let pred_new = insert_ml_run_clause_logical_op(currentDepth, replacementDepth, predicate, ml_model_retrieval, input_vars, output_var);
            return LogicalOperator::Projection { 
                predicate: Box::new(pred_new), 
                variables: variables.clone() 
            }
        }
        LogicalOperator::Selection { predicate, condition } => {
            let pred_new = insert_ml_run_clause_logical_op(currentDepth, replacementDepth, predicate, ml_model_retrieval, input_vars, output_var);
            return LogicalOperator::Selection { 
                predicate: Box::new(pred_new), 
                condition: condition.clone() 
            }
        }
        LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
            let input_new = insert_ml_run_clause_logical_op(currentDepth, replacementDepth, input, ml_model_retrieval, input_vars, output_var);
            return LogicalOperator::Bind { 
                input: Box::new(input_new), 
                function_name: function_name.clone(), 
                arguments: arguments.clone(), 
                output_variable: output_variable.clone() 
            }
        }
        LogicalOperator::Join { ref left, ref right } => {
            let left_new = insert_ml_run_clause_logical_op(currentDepth - 1, replacementDepth, left.as_ref(), ml_model_retrieval, input_vars, output_var);
            // let disp = format!("{left_new:?}");
            // println!("{disp} gets printed");
            return LogicalOperator::join(left_new, *right.clone())
        }
        _ => {logicalOp.clone()}
    }
}

// Helper function to convert pattern strings to TriplePattern
fn convert_pattern_to_triple(
    subject_str: &str,
    predicate_str: &str,
    object_str: &str,
    prefixes: &HashMap<String, String>,
    database: &mut SparqlDatabase,
) -> TriplePattern {
    let mut dict = database.dictionary.write().unwrap();
    
    let subject = if subject_str.starts_with('?') {
        Term::Variable(subject_str.to_string())
    } else {
        let resolved = resolve_with_prefixes(subject_str, prefixes);
        Term::Constant(dict.encode(&resolved))
    };

    let predicate = if predicate_str.starts_with('?') {
        Term::Variable(predicate_str.to_string())
    } else {
        let resolved = resolve_with_prefixes(predicate_str, prefixes);
        Term::Constant(dict.encode(&resolved))
    };

    let object = if object_str.starts_with('?') {
        Term::Variable(object_str.to_string())
    } else {
        let resolved = resolve_with_prefixes(object_str, prefixes);
        Term::Constant(dict.encode(&resolved))
    };
    
    drop(dict);

    (subject, predicate, object)
}

/// Builds a logical operator from a SubQuery structure
pub fn build_logical_plan_from_subquery(
    subquery: &SubQuery,
    prefixes: &HashMap<String, String>,
    database: &mut SparqlDatabase,
) -> LogicalOperator {
    // Build the inner logical plan from the subquery patterns
    let variables:  Vec<(&str, &str)> = subquery
        .variables
        .iter()
        .map(|(var_type, var_name, _aggregation)| (*var_type, *var_name))
        .collect();
    
    let inner_plan = build_logical_plan(
        variables.clone(),
        subquery.patterns.clone(),
        subquery.filters.clone(),
        prefixes,
        database,
        &subquery.binds,
        None,
        subquery.ml_run_clause.clone()
    );
    
    // Extract variable names for projection
    let projected_vars: Vec<String> = variables
        .iter()
        .map(|(_, var_name)| var_name.to_string())
        .collect();
    
    // Wrap in a subquery operator
    LogicalOperator::subquery(inner_plan, projected_vars)
}

/// Resolves a URI with prefixes
fn resolve_with_prefixes(uri: &str, prefixes: &HashMap<String, String>) -> String {
    if let Some(colon_pos) = uri.find(':') {
        let (prefix, suffix) = uri.split_at(colon_pos);
        if let Some(base_uri) = prefixes.get(prefix) {
            format!("{}{}", base_uri, &suffix[1..]) // Skip the ':'
        } else {
            uri.to_string()
        }
    } else {
        uri.to_string()
    }
}

// Helper function to process variables for aggregation
fn process_variables<'a>(
    selected_variables: &mut Vec<(String, String)>,
    aggregation_vars: &mut Vec<(&'a str, &'a str, &'a str)>,
    variables: Vec<(&'a str, &'a str, Option<&'a str>)>,
) {
    for (agg_type, var, opt_output_var) in variables {
        if agg_type == "SUM" || agg_type == "MIN" || agg_type == "MAX" || agg_type == "AVG" {
            let output_var = if let Some(name) = opt_output_var {
                name
            } else {
                ""
            };
            aggregation_vars.push((agg_type, var, output_var));
            selected_variables.push(("VAR".to_string(), output_var.to_string()));
        } else {
            selected_variables.push((agg_type.to_string(), var.to_string()));
        }
    }
}

/// Converts a FilterExpression with any lifetime to 'static lifetime
fn make_filter_static(filter: &FilterExpression) -> FilterExpression<'static> {
    match filter {
        FilterExpression::Comparison(var, op, value) => {
            let var_static: &'static str = Box::leak(var.to_string().into_boxed_str());
            let op_static: &'static str = Box::leak(op.to_string().into_boxed_str());
            let val_static: &'static str = Box::leak(value.to_string().into_boxed_str());
            FilterExpression::Comparison(var_static, op_static, val_static)
        }
        FilterExpression::And(left, right) => {
            FilterExpression::And(
                Box::new(make_filter_static(left)),
                Box::new(make_filter_static(right)),
            )
        }
        FilterExpression::Or(left, right) => {
            FilterExpression::Or(
                Box::new(make_filter_static(left)),
                Box::new(make_filter_static(right)),
            )
        }
        FilterExpression::Not(inner) => {
            FilterExpression::Not(Box::new(make_filter_static(inner)))
        }
        FilterExpression::ArithmeticExpr(expr) => {
            let expr_static: &'static str = Box::leak(expr.to_string().into_boxed_str());
            FilterExpression::ArithmeticExpr(expr_static)
        }
    }
}

/// Converts a FilterExpression to a Condition
fn convert_filter_to_condition(filter: &FilterExpression) -> Condition {
    // Convert the filter to have 'static lifetime by leaking strings
    let static_filter = make_filter_static(filter);
    Condition::from_filter(static_filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::terms::Term;

    #[test]
    fn test_pattern_contains_variable() {
        let pattern = (
            Term::Variable("s".to_string()),
            Term::Constant(1),
            Term::Variable("o".to_string()),
        );

        assert!(pattern_contains_variable(&pattern, "s"));
        assert!(pattern_contains_variable(&pattern, "o"));
        assert!(!pattern_contains_variable(&pattern, "p"));
    }

    #[test]
    fn test_count_bound_terms() {
        let pattern1 = (
            Term::Variable("s".to_string()),
            Term::Variable("p".to_string()),
            Term::Variable("o".to_string()),
        );
        assert_eq!(count_bound_terms(&pattern1), 0);

        let pattern2 = (
            Term::Constant(1),
            Term::Variable("p".to_string()),
            Term::Constant(2),
        );
        assert_eq!(count_bound_terms(&pattern2), 2);

        let pattern3 = (Term::Constant(1), Term::Constant(2), Term::Constant(3));
        assert_eq!(count_bound_terms(&pattern3), 3);
    }

    #[test]
    fn test_resolve_with_prefixes() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        let resolved = resolve_with_prefixes("ex:test", &prefixes);
        assert_eq!(resolved, "http://example.org/test");

        let unresolved = resolve_with_prefixes("http://other.org/test", &prefixes);
        assert_eq!(unresolved, "http://other.org/test");
    }

    #[test]
    fn test_ml_logical_plan_creation() {
        let database = &mut SparqlDatabase::new();
        let sparql = r#"PREFIX hvac: <http://example.org#>
        SELECT ?building ?energyPrediction WHERE {  
        ?building hvac:temperature ?temp.  
        ?building hvac:humidity ?humid.  
        ?building hvac:occupancy ?occ.  
        ?building hvac:sunlight ?sun.  
        ?building hvac:windSpeed ?wind.  
        ?building hvac:hour ?hour.  
        ?building hvac:dayOfWeek ?day.   
        RUN {?temp, ?humid, ?occ, ?sun, ?wind, ?hour, ?day} ON ?r TO ?energyPrediction.
        ?x hvac:hasValue ?energyPrediction.  
        ?building hvac:madeBy hvac:trumptower.
        }"#;

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

        parsed_prefixes.insert("hvac".to_string(), "https://housingass.org/measures".to_string());

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
            database, 
            &binds, 
            values_clause.as_ref(), 
            ml_run_clause.clone()
        );

        assert!(ml_run_clause.clone().is_some());
        let ml_run_clause_value = ml_run_clause.unwrap();

        let pattern1 = convert_pattern_to_triple(ml_run_clause_value.on, "rdf:type", "mls:Run", &parsed_prefixes, database);
        let pattern2 = convert_pattern_to_triple(ml_run_clause_value.on, "mls:hasOutput", "?mlmodel", &parsed_prefixes, database);
        let pattern3 = convert_pattern_to_triple("?mlmodel", "rdf:type", "mls:Model", &parsed_prefixes, database);

        let mut ml_retrieval_logical_op = LogicalOperator::scan(pattern1);
        ml_retrieval_logical_op = LogicalOperator::join (ml_retrieval_logical_op, LogicalOperator::scan(pattern2));
        ml_retrieval_logical_op = LogicalOperator::join(ml_retrieval_logical_op, LogicalOperator::scan(pattern3));
        ml_retrieval_logical_op = LogicalOperator::projection(ml_retrieval_logical_op, Vec::from(["?mlmodel".to_string()]));

        // let mut expected_logical_operator = LogicalOperator::scan(
        //         convert_pattern_to_triple(
        //         "?building", 
        //         "hvac:month", 
        //         "hvac:february", 
        //         &parsed_prefixes, 
        //         database
        //         )
        //     );
        let mut expected_logical_operator = LogicalOperator::scan(
            convert_pattern_to_triple(
                "?building", 
                "hvac:temperature", 
                "?temp", 
                &parsed_prefixes, 
                database
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:humidity", 
                "?humid", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:occupancy", 
                "?occ", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:sunlight", 
                "?sun", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:windSpeed", 
                "?wind", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:hour", 
                "?hour", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:dayOfWeek", 
                "?day", 
                &parsed_prefixes, 
                database
                )
            )
        );

        let input_vec = ml_run_clause_value.run.clone().iter().map(|x| x.to_string()).collect();
        let output_str = ml_run_clause_value.to.to_string();
        expected_logical_operator = LogicalOperator::run_ml_clause_lo(expected_logical_operator, ml_retrieval_logical_op, input_vec, output_str);

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?x", "hvac:hasValue", "?energyPrediction", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?building", "hvac:madeBy", "hvac:trumptower", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::projection(expected_logical_operator, Vec::from(["?building".to_string(), "?energyPrediction".to_string()]));

        assert_eq!(format!("{expected_logical_operator:?}"), format!("{produced_logical_operator:?}"));

    }

    #[test]
    fn test_ml_logical_plan_creation_with_ml() {
        let database = &mut SparqlDatabase::new();
        let sparql = r#"PREFIX hvac: <http://example.org#>
        SELECT ?building ?energyPrediction WHERE {  
        ?building hvac:temperature ?temp.  
        ?building hvac:humidity ?humid.  
        ?building hvac:occupancy ?occ.  
        ?building hvac:sunlight ?sun.  
        ?building hvac:windSpeed ?wind.  
        ?building hvac:hour ?hour.  
        ?building hvac:dayOfWeek ?day.
        ?r rdf:type mls:Run.
        ?r mls:hasOutput ?ml.
        ?ml rdf:type mls:Model.   
        RUN {?temp, ?humid, ?occ, ?sun, ?wind, ?hour, ?day} ON ?r TO ?energyPrediction.
        ?x hvac:hasValue ?energyPrediction.  
        ?building hvac:madeBy hvac:trumptower.
        }"#;

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

        parsed_prefixes.insert("hvac".to_string(), "https://housingass.org/measures".to_string());

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
            database, 
            &binds, 
            values_clause.as_ref(), 
            ml_run_clause.clone()
        );

        assert!(ml_run_clause.clone().is_some());
        let ml_run_clause_value = ml_run_clause.unwrap();

        // let mut expected_logical_operator = LogicalOperator::scan(
        //         convert_pattern_to_triple(
        //         "?building", 
        //         "hvac:month", 
        //         "hvac:february", 
        //         &parsed_prefixes, 
        //         database
        //         )
        //     );
        let mut expected_logical_operator = LogicalOperator::scan(
            convert_pattern_to_triple(
                "?building", 
                "hvac:temperature", 
                "?temp", 
                &parsed_prefixes, 
                database
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:humidity", 
                "?humid", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:occupancy", 
                "?occ", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:sunlight", 
                "?sun", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:windSpeed", 
                "?wind", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:hour", 
                "?hour", 
                &parsed_prefixes, 
                database
                )
            )
        );
        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple(
                "?building", 
                "hvac:dayOfWeek", 
                "?day", 
                &parsed_prefixes, 
                database
                )
            )
        );

        let mut ml_retrieval_logical_op = expected_logical_operator.clone();
        ml_retrieval_logical_op = LogicalOperator::join(
            ml_retrieval_logical_op, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?r", "rdf:type", "mls:Run", &parsed_prefixes, database)
            )
        );

        ml_retrieval_logical_op = LogicalOperator::join(
            ml_retrieval_logical_op, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?r", "mls:hasOutput", "?ml", &parsed_prefixes, database)
            )
        );

        ml_retrieval_logical_op = LogicalOperator::join(
            ml_retrieval_logical_op, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?ml", "rdf:type", "mls:Model", &parsed_prefixes, database)
            )
        );

        ml_retrieval_logical_op = LogicalOperator::projection(ml_retrieval_logical_op, Vec::from(["?ml".to_string()]));

        let input_vec = ml_run_clause_value.run.clone().iter().map(|x| x.to_string()).collect();
        let output_str = ml_run_clause_value.to.to_string();

        expected_logical_operator = LogicalOperator::run_ml_clause_lo(expected_logical_operator, ml_retrieval_logical_op, input_vec, output_str);

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?r", "rdf:type", "mls:Run", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?r", "mls:hasOutput", "?ml", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?ml", "rdf:type", "mls:Model", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?x", "hvac:hasValue", "?energyPrediction", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::join(
            expected_logical_operator, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?building", "hvac:madeBy", "hvac:trumptower", &parsed_prefixes, database)
            )
        );

        expected_logical_operator = LogicalOperator::projection(expected_logical_operator, Vec::from(["?building".to_string(), "?energyPrediction".to_string()]));

        assert_eq!(format!("{expected_logical_operator:?}"), format!("{produced_logical_operator:?}"));

    }
}
