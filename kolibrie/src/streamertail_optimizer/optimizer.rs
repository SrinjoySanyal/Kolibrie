/*
 * Copyright © 2024 Volodymyr Kadzhaia
 * Copyright © 2024 Pieter Bonte
 * KU Leuven — Stream Intelligence Lab, Belgium
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * you can obtain one at https://mozilla.org/MPL/2.0/.
 */

use super::cost::CostEstimator;
use super::execution::ExecutionEngine;
use super::operators::{LogicalOperator, PhysicalOperator};
use super::stats::DatabaseStats;

use crate::sparql_database::SparqlDatabase;
use crate::streamertail_optimizer::operators::logical::ModelGetter;
use crate::streamertail_optimizer::operators::physical::ModelGetterPhysical;
use shared::terms::{Term, TriplePattern};
use shared::triple::Triple;
use shared::query::FilterExpression;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::format;
use std::sync::Arc;

/// Volcano-style query optimizer with cost-based optimization
pub struct Streamertail {
    pub memo: HashMap<String, PhysicalOperator>,
    pub selected_variables: Vec<String>,
    pub stats: Arc<DatabaseStats>,
    pub mlmemo: HashMap<(BTreeSet<Triple>, String), Vec<HashMap<String, u32>>>
}

fn serialize_arith_expr(expr: &shared::query::ArithmeticExpression) -> String {
    use shared::query::ArithmeticExpression as AE;
    match expr {
        AE::Operand(s) => s.to_string(),
        AE::Add(l, r) => format!("({} + {})", serialize_arith_expr(l), serialize_arith_expr(r)),
        AE::Subtract(l, r) => format!("({} - {})", serialize_arith_expr(l), serialize_arith_expr(r)),
        AE::Multiply(l, r) => format!("({} * {})", serialize_arith_expr(l), serialize_arith_expr(r)),
        AE::Divide(l, r) => format!("({} / {})", serialize_arith_expr(l), serialize_arith_expr(r)),
    }
}

impl Streamertail {
    /// Creates a new volcano optimizer
    pub fn new(database: &SparqlDatabase) -> Self {
        let stats = Arc::new(DatabaseStats::gather_stats_fast(database));
        Self {
            memo: HashMap::new(),
            selected_variables: Vec::new(),
            stats,
            mlmemo: HashMap::new()
        }
    }

    pub fn with_cached_stats(stats: Arc<DatabaseStats>) -> Self {
        Self {
            memo: HashMap::new(),
            selected_variables: Vec::new(),
            stats,
            mlmemo: HashMap::new()
        }
    }

    /// Finds the best physical plan for a logical plan
    pub fn find_best_plan(&mut self, logical_plan: &LogicalOperator, database: &SparqlDatabase) -> PhysicalOperator {
        self.find_best_plan_recursive(logical_plan, database)
    }

    /// Executes a physical plan and returns results
    pub fn execute_plan(
        &self,
        plan: &PhysicalOperator,
        database: &mut SparqlDatabase,
    ) -> Vec<HashMap<String, String>> {
        ExecutionEngine::execute(plan, database)
    }

    /// Optimizes and executes a logical plan in one step
    pub fn optimize_and_execute(
        &mut self,
        logical_plan: &LogicalOperator,
        database: &mut SparqlDatabase,
    ) -> Vec<HashMap<String, String>> {
        let physical_plan = self.find_best_plan(logical_plan, database);
        self.execute_plan(&physical_plan, database)
    }

    /// Detects if a join tree is a star query pattern
    fn is_star_query(&self, plan: &LogicalOperator) -> Option<Vec<(String, Vec<TriplePattern>)>> {
        let mut patterns = Vec::new();
        self.collect_patterns(plan, &mut patterns);

        if patterns.len() < 3 {
            return None;
        }

        // Count how many patterns each variable appears
        let mut var_counts: std::collections::BTreeMap<String, Vec<usize>> = BTreeMap::new();

        for (idx, pattern) in patterns.iter().enumerate() {
            if let Term::Variable(var) = &pattern.0 {
                var_counts.entry(var.clone()).or_default().push(idx);
            }
            if let Term::Variable(var) = &pattern.1 {
                var_counts.entry(var.clone()).or_default().push(idx);
            }
            if let Term::Variable(var) = &pattern.2 {
                var_counts.entry(var.clone()).or_default().push(idx);
            }
        }

        // Find all variables that appear in 2+ patterns
        let mut star_vars: Vec<(&String, &Vec<usize>)> = var_counts
            .iter()
            .filter(|(_, indices)| indices.len() >= 2)  // <- Lowered from 3 to 2
            .collect();

        // Sort by number of occurrences (most frequent first)
        star_vars.sort_by_key(|(_, indices)| std::cmp::Reverse(indices.len()));

        if star_vars.is_empty() {
            return None;
        }

        // Greedily assign patterns to stars
        let mut used_patterns: HashSet<usize> = HashSet::new();
        let mut stars: Vec<(String, Vec<TriplePattern>)> = Vec::new();

        for (var, pattern_indices) in star_vars {
            // Get patterns for this variable that haven't been used yet
            let available: Vec<usize> = pattern_indices
                .iter()
                .filter(|&&idx| !used_patterns.contains(&idx))
                .copied()
                .collect();

            if available.len() >= 2 {  // Need at least 2 patterns for a star
                let star_patterns: Vec<TriplePattern> = available
                    .iter()
                    .map(|&idx| patterns[idx].clone())
                    .collect();

                // Mark these patterns as used
                for &idx in &available {
                    used_patterns.insert(idx);
                }

                stars.push((var.clone(), star_patterns));
            }
        }

        if stars.is_empty() {
            None
        } else {
            Some(stars)
        }
    }

    fn has_ml_predict(&self, plan: &LogicalOperator) -> bool {
        match plan {
            LogicalOperator::Scan { pattern } => {
                false
            }
            LogicalOperator::Join { left, right } => {
                self.has_ml_predict(left) || self.has_ml_predict(right)
            }
            LogicalOperator::Selection { predicate, ..  } => {
                self.has_ml_predict(predicate)
            }
            LogicalOperator::Projection { predicate, .. } => {
                self.has_ml_predict(predicate)
            }
            LogicalOperator::Buffer { content: _, origin: _ } => { false }
            LogicalOperator::Subquery { inner, .. } => {
                // Subqueries are treated as separate scopes, so we don't collect their patterns
                // for star query detection in the outer query
                self.has_ml_predict(inner)
            }
            LogicalOperator::Bind { input, .. } => {
                self.has_ml_predict(input)
            }
            LogicalOperator::Values { .. } => { false }
            LogicalOperator::MLPredict { input, model_name, .. } => {
                match model_name {
                    ModelGetter::MLPredict(..) => {
                        self.has_ml_predict(input)
                    }
                    ModelGetter::RunMLClause(..) => {true}
                }
            }
        }
    }

    fn collect_patterns(&self, plan: &LogicalOperator, patterns: &mut Vec<TriplePattern>) {
        match plan {
            LogicalOperator::Scan { pattern } => {
                patterns.push(pattern.clone());
            }
            LogicalOperator::Join { left, right } => {
                if !self.has_ml_predict(left.as_ref()){
                    self.collect_patterns(left, patterns);
                }
                if !self.has_ml_predict(right.as_ref()){
                    self.collect_patterns(right, patterns);
                }
            }
            LogicalOperator::Selection { predicate, ..  } => {
                self.collect_patterns(predicate, patterns);
            }
            LogicalOperator::Projection { predicate, .. } => {
                self.collect_patterns(predicate, patterns);
            }
            LogicalOperator::Buffer { content: _, origin: _ } => { }
            LogicalOperator::Subquery { inner, .. } => {
                // Subqueries are treated as separate scopes, so we don't collect their patterns
                // for star query detection in the outer query
                self.collect_patterns(inner, patterns);
            }
            LogicalOperator::Bind { input, .. } => {
                self.collect_patterns(input, patterns);
            }
            LogicalOperator::Values { .. } => { }
            LogicalOperator::MLPredict { input, model_name, .. } => {
                match model_name {
                    ModelGetter::MLPredict(..) => {
                        self.collect_patterns(input, patterns);
                    }
                    ModelGetter::RunMLClause(..) => {patterns.clear();}
                }
            }
        }
    }

    fn bubble_up_runmlclause(
        &self,
        logicalOp: &LogicalOperator,
        mlinput: &LogicalOperator,
        mlretrieval: &LogicalOperator,
        input_variables: Vec<String>, 
        output_var: String
    ) -> LogicalOperator {
        match logicalOp {
            LogicalOperator::Join { left, right } => {
                let removed_ml = LogicalOperator::join(self.get_lo_without_ml(left), self.get_lo_without_ml(right));
                if (self.estimate_logical_cost(logicalOp) < self.estimate_logical_cost(mlinput)){
                    return LogicalOperator::run_ml_clause_lo(logicalOp.clone(), mlinput.clone(), input_variables, output_var)
                }
                // This function needs to be applied before a Join reordering, meaning that the joins are left-deep
                return self.bubble_up_runmlclause(left, mlinput, mlretrieval, input_variables, output_var)
            }
            LogicalOperator::Scan {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Buffer {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Selection { predicate, condition } => {
                return LogicalOperator::selection(self.bubble_up_runmlclause(predicate, mlinput, mlretrieval, input_variables, output_var), condition.clone());
            }
            LogicalOperator::MLPredict {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Projection { predicate, variables } => {
                return LogicalOperator::projection(self.bubble_up_runmlclause(predicate, mlinput, mlretrieval, input_variables, output_var), variables.clone())
            }
            LogicalOperator::Subquery { inner, projected_vars } => {
                return LogicalOperator::subquery(self.bubble_up_runmlclause(inner.as_ref(), mlinput, mlretrieval, input_variables, output_var), projected_vars.clone());
            }
            LogicalOperator::Values {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
                return LogicalOperator::bind(self.bubble_up_runmlclause(input.as_ref(), mlinput, mlretrieval, input_variables, output_var), function_name.clone(), arguments.clone(), output_variable.clone())
            }
        }
    }

    fn get_lo_without_ml(&self, logicalOp: &LogicalOperator) -> LogicalOperator {
        match logicalOp {
            LogicalOperator::Join { left, right } => {
                return LogicalOperator::join(self.get_lo_without_ml(left), self.get_lo_without_ml(right));
            }
            LogicalOperator::MLPredict { input, model_name, input_variables, output_variable } => {
                match model_name {
                    ModelGetter::RunMLClause(lo) => {
                        return lo.as_ref().clone();
                    }
                    ModelGetter::MLPredict(name) => {
                        return logicalOp.clone();
                    }
                }
            }
            LogicalOperator::Scan {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Buffer {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Selection {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Projection {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Subquery {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Values {..} => {
                return logicalOp.clone();
            }
            LogicalOperator::Bind {..} => {
                return logicalOp.clone();
            }
        }
    }

    fn ml_exists(&self, logicalOp: &LogicalOperator) -> Option<(LogicalOperator, LogicalOperator, Vec<String>, String)> {
        match logicalOp {
            LogicalOperator::MLPredict { input, model_name, input_variables, output_variable } => {
                match model_name {
                    ModelGetter::RunMLClause(op) => {
                        return Some((logicalOp.clone(), op.as_ref().clone(), input_variables.clone(), output_variable.clone()));
                    }
                    _ => {return None;}
                }
            }
            LogicalOperator::Projection { predicate, variables } => {
                self.ml_exists(predicate.as_ref())
            }
            LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
                self.ml_exists(input.as_ref())
            }
            LogicalOperator::Selection { predicate, condition } => {
                self.ml_exists(predicate.as_ref())
            }
            // ml_exists is called before any query optimization, so the query is initially left-deep
            LogicalOperator::Join { left, right } => {
                self.ml_exists(left.as_ref())
            }
            LogicalOperator::Subquery { inner, projected_vars } => {
                self.ml_exists(inner.as_ref())
            }
            _ => None
        }
    }

    pub fn find_best_plan_recursive_optimised(&mut self, logical_plan: &LogicalOperator, database: &SparqlDatabase) -> PhysicalOperator {
        let mut logical_plan_clone = logical_plan.clone();
        if let Some((mlinput, mlretrieval, input_vars, output_var)) = self.ml_exists(logical_plan){
            logical_plan_clone = self.bubble_up_runmlclause(logical_plan, &mlinput, &mlretrieval, input_vars, output_var);
        }

        self.find_best_plan_recursive(&logical_plan_clone, database)
    }

    /// Recursively finds the best plan using dynamic programming with memoization
    fn find_best_plan_recursive(&mut self, logical_plan: &LogicalOperator, database: &SparqlDatabase) -> PhysicalOperator {
        let key = self.create_memo_key(logical_plan);

        if let Some(plan) = self.memo.get(&key) {
            return plan.clone();
        }

        if let LogicalOperator::Projection { predicate: proj_pred, variables } = logical_plan {
            if let LogicalOperator::Selection { predicate: sel_pred, condition } = proj_pred.as_ref() {
                if let Some(stars) = self.is_star_query(sel_pred) {
                    // Build: Projection(Filter(StarJoin))
                    let star_plan = self.build_star_join_from_patterns(stars, sel_pred);
                    let filtered_plan = PhysicalOperator::filter(star_plan, condition.clone());
                    let projected_plan = PhysicalOperator::projection(filtered_plan, variables.clone());
                    self.memo.insert(key, projected_plan.clone());
                    return projected_plan;
                }
            }
        }

        // Handle Selection wrapping star query (no projection)
        if let LogicalOperator::Selection { predicate, condition } = logical_plan {
            if let Some(stars) = self.is_star_query(predicate) {
                let star_plan = self.build_star_join_from_patterns(stars, predicate);
                let filtered_plan = PhysicalOperator::filter(star_plan, condition.clone());
                self.memo.insert(key, filtered_plan.clone());
                return filtered_plan;
            }
        }

        // Handle star query without selection or projection
        if ! matches!(logical_plan, LogicalOperator::Selection { .. } | LogicalOperator::Projection { ..  }) {
            if let Some(stars) = self.is_star_query(logical_plan) {
                let star_plan = self.build_star_join_from_patterns(stars, logical_plan);
                self.memo.insert(key, star_plan.clone());
                return star_plan;
            }
        }

        let mut candidates = Vec::new();

        match logical_plan {
            LogicalOperator::Scan { pattern } => {
                // Implementation rules: Map logical scan to physical scans
                let best_scan = self.choose_best_scan(pattern);
                candidates.push(best_scan);
            }
            LogicalOperator::Selection {
                predicate,
                condition,
            } => {
                // Transformations: Push down selections
                let best_child_plan = self.find_best_plan_recursive(predicate, database);
                // Implementation rules: Apply selection as a filter
                candidates.push(PhysicalOperator::filter(best_child_plan, condition.clone()));
            }
            LogicalOperator::Projection {
                predicate,
                variables,
            } => {
                let best_child_plan = self.find_best_plan_recursive(predicate, database);
                candidates.push(PhysicalOperator::projection(
                    best_child_plan,
                    variables.clone(),
                ));
            }
            LogicalOperator::Join { left, right } => {
                // Add join reordering based on cost
                let left_cost = self.estimate_logical_cost(left);
                let right_cost = self.estimate_logical_cost(right);

                let (cheaper_side, expensive_side) = if left_cost <= right_cost {
                    (left, right)
                } else {
                    (right, left) // Swap for better order
                };

                let best_left_plan = self.find_best_plan_recursive(cheaper_side, database);
                let best_right_plan = self.find_best_plan_recursive(expensive_side, database);

                // Implementation rules: Different join algorithms
                candidates.push(PhysicalOperator::optimized_hash_join(
                    best_left_plan.clone(),
                    best_right_plan.clone(),
                ));

                candidates.push(PhysicalOperator::hash_join(
                    best_left_plan.clone(),
                    best_right_plan.clone(),
                ));

                // Only use nested loop for small datasets
                let left_cardinality = self.estimate_output_cardinality_from_logical(cheaper_side);
                let right_cardinality =
                    self.estimate_output_cardinality_from_logical(expensive_side);

                if left_cardinality < 1000 && right_cardinality < 1000 {
                    candidates.push(PhysicalOperator::nested_loop_join(
                        best_left_plan.clone(),
                        best_right_plan.clone(),
                    ));
                }

                // Add parallel join option
                candidates.push(PhysicalOperator::parallel_join(
                    best_left_plan,
                    best_right_plan,
                ));
            }
            LogicalOperator::Buffer { content, origin} => {
                let best_buffer = PhysicalOperator::InMemoryBuffer {content: content.clone(), origin: origin.clone()};
                candidates.push(best_buffer);
            }
            LogicalOperator::Subquery { inner, projected_vars } => {
                // Recursively optimize the inner query
                let optimized_inner = self.find_best_plan_recursive(inner, database);
                
                // Wrap it in a subquery operator with projection
                let subquery_plan = PhysicalOperator::subquery(
                    optimized_inner,
                    projected_vars.clone()
                );
                
                candidates.push(subquery_plan);
            }
            LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
                // Recursively optimize the input
                let best_input_plan = self.find_best_plan_recursive(input, database);
    
                // Create the physical BIND operator
                let bind_plan = PhysicalOperator::bind(
                    best_input_plan,
                    function_name.clone(),
                    arguments.clone(),
                    output_variable.clone(),
                );
    
                candidates.push(bind_plan);
            }
            LogicalOperator::Values { variables, values } => {
                // VALUES is a leaf operator
                candidates.push(PhysicalOperator::values(
                    variables.clone(),
                    values.clone(),
                ));
            }
            LogicalOperator::MLPredict {
                input,
                model_name,
                input_variables,
                output_variable,
            } => {
                // Recursively optimize the input
                let best_input_plan = self.find_best_plan_recursive(input, database);

                // Discover model path
                let model_path = self.discover_model_path();
                // println!("model path = {model_path}");

                // let mut modelstr;

                match model_name {
                    ModelGetter::MLPredict(mlstring) => {
                        let modelstr = mlstring;
                        // Create the physical ML.PREDICT operator
                        let ml_predict_plan = PhysicalOperator::ml_predict(
                            best_input_plan,
                            modelstr.clone(),
                            model_path,
                            input_variables.clone(),
                            output_variable.clone(),
                        );

                        candidates.push(ml_predict_plan);
                    }
                    ModelGetter::RunMLClause(argument) => {
                        let models = ExecutionEngine::execute_with_ids(&self.find_best_plan_recursive(argument.as_ref(), database), &mut database.clone());
                        // self.mlmemo.insert((database.triples.clone(), physical_operator_ml_string.clone()), models.clone());
                    
                        // Vector of HashMaps which have a single key-value pair
                        // Each of the HashMaps have the same key: the variable storing the machine learning models
                        let namelist: Vec<HashMap<String, String>> = models.clone().into_iter()
                        .map(|id_result| {
                            let dict = database.dictionary.read().unwrap();
                            let result = id_result
                            .into_iter()
                            // maps the id of a model name value to the model value itself
                            .map(|(var, id)| (var, dict.decode(id).unwrap().to_string()))
                            .collect();
                            drop(dict);
                            result
                        })
                        .collect();
                        println!("[RUN_ML_CLAUSE] Models retrieved: {namelist:?}");

                        let ml_predict_plan = PhysicalOperator::run_ml_clause(
                            best_input_plan,
                            models,
                            namelist,
                            model_path,
                            input_variables.clone(),
                            output_variable.clone(),
                        );

                        candidates.push(ml_predict_plan);
                    }
                }
            }
        }

        // Cost-based optimization: Choose the best candidate
        let cost_estimator = CostEstimator::new(&self.stats);
        let best_plan = candidates
            .into_iter()
            .min_by_key(|plan| {
                let cost = cost_estimator.estimate_cost(plan);
                cost
            })
            .unwrap();

        // Memoize the best plan
        self.memo.insert(key, best_plan.clone());
        best_plan
    }

    /// Discovers the model path from the model name
    fn discover_model_path(&self) -> String {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        
        loop {
            let ml_dir = path.join("ml");
            if ml_dir.exists() && ml_dir.is_dir() {
                let model_dir = ml_dir.join("examples").join("models");
                
                // Return just the model directory - the ML handler will discover models
                if model_dir.exists() {
                    return model_dir.to_string_lossy().to_string();
                }
                
                break;
            }
            
            if !path.pop() {
                eprintln!("Warning: Could not locate 'ml' directory!");
                break;
            }
        }
        
        // Fallback to relative path
        format!("ml/examples/models")
    }

    /// Helper method to build a star join physical plan from detected star patterns
    fn build_star_join_from_patterns(
        &mut self,
        stars: Vec<(String, Vec<TriplePattern>)>,
        logical_plan: &LogicalOperator,
    ) -> PhysicalOperator {
        let mut all_patterns = Vec::new();
        self.collect_patterns(logical_plan, &mut all_patterns);

        let mut used_pattern_indices: HashSet<usize> = HashSet::new();
        for (_, star_patterns) in &stars {
            for star_pattern in star_patterns {
                if let Some(idx) = all_patterns.iter().position(|p| p == star_pattern) {
                    used_pattern_indices.insert(idx);
                }
            }
        }

        if stars.len() > 1 {
            let mut star_operators: Vec<(String, Vec<TriplePattern>)> = stars;

            star_operators.sort_by_key(|(_, patterns)| {
                let bound_count = patterns.iter().filter(|p| {
                    matches!(p.0, Term::Constant(_)) ||
                    matches!(p.1, Term::Constant(_)) ||
                    matches!(p.2, Term::Constant(_))
                }).count();
                std::cmp::Reverse(bound_count)
            });

            let (first_var, first_patterns) = star_operators.remove(0);
            let mut result = PhysicalOperator::StarJoin {
                join_var: first_var.clone(),
                patterns: first_patterns,
            };

            for (_, patterns) in star_operators {
                let star_scans: Vec<PhysicalOperator> = patterns
                    .into_iter()
                    .map(|pattern| PhysicalOperator::index_scan(pattern))
                    .collect();

                for scan in star_scans {
                    result = PhysicalOperator::parallel_join(result, scan);
                }
            }

            for (idx, pattern) in all_patterns.iter().enumerate() {
                if !used_pattern_indices.contains(&idx) {
                    let scan = PhysicalOperator::index_scan(pattern.clone());
                    result = PhysicalOperator::parallel_join(result, scan);
                }
            }

            result
        } else if stars.len() == 1 {
            let (join_var, patterns) = stars.into_iter().next().unwrap();

            if used_pattern_indices.len() < all_patterns.len() {
                let mut result = PhysicalOperator::StarJoin { join_var, patterns };

                for (idx, pattern) in all_patterns.iter().enumerate() {
                    if !used_pattern_indices.contains(&idx) {
                        let scan = PhysicalOperator::index_scan(pattern.clone());
                        result = PhysicalOperator::parallel_join(result, scan);
                    }
                }

                result
            } else {
                PhysicalOperator::StarJoin { join_var, patterns }
            }
        } else {
            // Shouldn't happen, but return a dummy scan as fallback
            PhysicalOperator::table_scan((
                Term::Variable("?s".to_string()),
                Term::Variable("?p".to_string()),
                Term::Variable("?o".to_string()),
            ))
        }
    }

    /// Chooses the best scan method based on pattern selectivity
    fn choose_best_scan(&self, pattern: &TriplePattern) -> PhysicalOperator {
        let bound_vars = self.count_bound_variables(pattern);
        let cost_estimator = CostEstimator::new(&self.stats);
        let estimated_size = cost_estimator.estimate_cardinality(pattern);

        match bound_vars {
            3 => PhysicalOperator::index_scan(pattern.clone()), // Fully bound - always use index
            2 => PhysicalOperator::index_scan(pattern.clone()), // Two bounds - index is better
            1 => {
                // Use index if result set is small enough
                if estimated_size < 10000 {
                    PhysicalOperator::index_scan(pattern.clone())
                } else {
                    PhysicalOperator::table_scan(pattern.clone())
                }
            }
            0 => PhysicalOperator::table_scan(pattern.clone()), // Full scan
            _ => PhysicalOperator::table_scan(pattern.clone()),
        }
    }

    /// Counts the number of bound variables in a triple pattern
    fn count_bound_variables(&self, pattern: &TriplePattern) -> usize {
        let mut count = 0;

        match &pattern.0 {
            Term::Constant(_) => count += 1,
            Term::Variable(_) | Term::QuotedTriple(_) => {}
        }

        match &pattern.1 {
            Term::Constant(_) => count += 1,
            Term::Variable(_) | Term::QuotedTriple(_) => {}
        }

        match &pattern.2 {
            Term::Constant(_) => count += 1,
            Term::Variable(_) | Term::QuotedTriple(_) => {}
        }

        count
    }

    /// Creates a memo key for caching optimized plans
    fn create_memo_key(&self, logical_plan: &LogicalOperator) -> String {
        self.serialize_logical_plan(logical_plan)
    }

    /// Serializes a physical plan to a string for memoization
    fn serialize_physical_plan(&self, plan: &PhysicalOperator) -> String {
        match plan {
            PhysicalOperator::Bind { input, function_name, arguments, output_variable } => {
                format!("Bind([{:?}],{:?},{:?},{:?})", self.serialize_physical_plan(input.as_ref()), function_name, arguments, output_variable)
            }
            PhysicalOperator::Filter { input, condition } => {
                format!(
                    "Filter([{}], {})",
                    self.serialize_physical_plan(input),
                    self.serialize_filter_expression(&condition.expression)
                )
            }
            PhysicalOperator::Projection {
                input,
                variables,
            } => {
                format!(
                    "Projection({:?},[{}])",
                    variables,
                    self.serialize_physical_plan(input)
                )
            }
            PhysicalOperator::HashJoin { left, right } => {
                format!(
                    "HashJoin([{:?}], [{:?}])", 
                    self.serialize_physical_plan(left),
                    self.serialize_physical_plan(right)
                )
            }
            PhysicalOperator::OptimizedHashJoin { left, right } => {
                format!(
                    "OptimizedHashJoin([{:?}], [{:?}])", 
                    self.serialize_physical_plan(left),
                    self.serialize_physical_plan(right)
                )
            }
            PhysicalOperator::NestedLoopJoin { left, right } => {
                format!(
                    "NestedJoin([{:?}], [{:?}])", 
                    self.serialize_physical_plan(left),
                    self.serialize_physical_plan(right)
                )
            }
            PhysicalOperator::ParallelJoin { left, right } => {
                format!(
                    "ParallelJoin([{:?}], [{:?}])", 
                    self.serialize_physical_plan(left),
                    self.serialize_physical_plan(right)
                )
            }
            PhysicalOperator::StarJoin { join_var, patterns } => {
                format!(
                    "StarJoin({:?}, {:?})", 
                    join_var,
                    patterns
                )
            }
            PhysicalOperator::InMemoryBuffer { content, origin } => {
                format!("InMemoryBuffer({:?}, {:?})", content, origin)
            }
            PhysicalOperator::IndexScan { pattern } => {
                format!("IndexScan({:?})", pattern)
            }
            PhysicalOperator::MLPredict { input, model_name, model_path, input_variables, output_variable } => {
                let modelstr = match model_name {
                    ModelGetterPhysical::MLPredictPhysical(mlstring) => {
                        mlstring
                    }
                    ModelGetterPhysical::RunMLClausePhysical(mlid, ml) => {
                        &format!("RunMLClausePhysical({:?}, {:?})", mlid, ml)
                    }
                };
                format!(
                    "MLPredict([{:?}], {:?}, {:?}, {:?}, {:?})",
                    self.serialize_physical_plan(input),
                    modelstr,
                    model_path,
                    input_variables,
                    output_variable
                )
            }
            PhysicalOperator::TableScan { pattern } => {
                format!("TableScan({:?})", pattern)
            }
            PhysicalOperator::Values { variables, values } => {
                format!("Values({:?}, {:?})", variables, values)
            }
            PhysicalOperator::Subquery { inner, projected_vars } => {
                format!("Subquery([{:?}], {:?})", self.serialize_physical_plan(inner), projected_vars)
            }
        }
    }

    /// Serializes a logical plan to a string for memoization
    fn serialize_logical_plan(&self, plan: &LogicalOperator) -> String {
        match plan {
            LogicalOperator::Scan { pattern } => {
                format!("Scan({:?},{:?},{:?})", pattern.0, pattern.1, pattern.2)
            }
            LogicalOperator::Selection {
                predicate,
                condition,
            } => {
                format!(
                    "Selection([{}], {})",
                    self.serialize_logical_plan(predicate),
                    self.serialize_filter_expression(&condition.expression)
                )
            }
            LogicalOperator::Projection {
                predicate,
                variables,
            } => {
                format!(
                    "Projection({:?},[{}])",
                    variables,
                    self.serialize_logical_plan(predicate)
                )
            }
            LogicalOperator::Join { left, right } => {
                format!(
                    "Join([{}],[{}])",
                    self.serialize_logical_plan(left),
                    self.serialize_logical_plan(right)
                )
            }
            LogicalOperator::Buffer { content, origin } => {
                format!(
                    "Buffer({:?},{:?})",
                    origin,
                    content
                )
            }
            LogicalOperator::Subquery { inner, projected_vars } => {
                format!(
                    "Subquery({:?},[{}])",
                    projected_vars,
                    self.serialize_logical_plan(inner)
                )
            }
            LogicalOperator::Bind { input, function_name, arguments, output_variable } => {
                format!(
                    "Bind({}, {}({:?}), {})",
                    self.serialize_logical_plan(input),
                    function_name,
                    arguments,
                    output_variable
                )
            }
            LogicalOperator::Values { variables, values } => {
                format!(
                    "Values({:?}, {} rows)",
                    variables,
                    values.len()
                )
            }
            LogicalOperator::MLPredict {
                input,
                model_name,
                input_variables,
                output_variable,
            } => {
                let modelstr = match model_name {
                    ModelGetter::MLPredict(mlstring) => {
                        mlstring
                    }
                    ModelGetter::RunMLClause(argument) => {
                        &self.serialize_logical_plan(argument)
                    }
                };
                format!(
                    "MLPredict({}, model={}, inputs={:?}, output={})",
                    self.serialize_logical_plan(input),
                    modelstr.clone(),
                    input_variables,
                    output_variable
                )
            }
        }
    }

    /// Serializes a filter expression to a string
    fn serialize_filter_expression(&self, expr: &FilterExpression) -> String {
        match expr {
            FilterExpression::Comparison(var, op, value) => {
                format!("{}{}'{}'", var, op, value)
            }
            FilterExpression::And(left, right) => {
                format!(
                    "({} AND {})",
                    self.serialize_filter_expression(left),
                    self.serialize_filter_expression(right)
                )
            }
            FilterExpression::Or(left, right) => {
                format!(
                    "({} OR {})",
                    self.serialize_filter_expression(left),
                    self.serialize_filter_expression(right)
                )
            }
            FilterExpression::Not(inner) => {
                format!("NOT({})", self.serialize_filter_expression(inner))
            }
            FilterExpression::ArithmeticExpr(expr) => {
                format!("ARITH({})", serialize_arith_expr(expr))
            }
            FilterExpression::FunctionCall(name, args) => {
                format!("{}({})", name, args.join(", "))
            }
        }
    }

    /// Estimates the cost of a logical plan
    pub fn estimate_logical_cost(&self, logical_plan: &LogicalOperator) -> u64 {
        let cost_estimator = CostEstimator::new(&self.stats);

        match logical_plan {
            LogicalOperator::Scan { pattern } => cost_estimator.estimate_cardinality(pattern),
            LogicalOperator::Join { left, right } => {
                let left_cost = self.estimate_logical_cost(left);
                let right_cost = self.estimate_logical_cost(right);
                let left_card = self.estimate_output_cardinality_from_logical(left);
                let right_card = self.estimate_output_cardinality_from_logical(right);

                // More sophisticated join cost estimation
                let join_selectivity = self.estimate_join_selectivity(left, right);
                left_cost + right_cost + ((left_card * right_card) as f64 * join_selectivity) as u64
            }
            LogicalOperator::Selection {
                predicate,
                condition,
            } => {
                let base_cost = self.estimate_logical_cost(predicate);
                let selectivity = cost_estimator.estimate_selectivity(condition);
                (base_cost as f64 * selectivity) as u64
            }
            LogicalOperator::Projection { predicate, .. } => self.estimate_logical_cost(predicate),
            LogicalOperator::Buffer { .. } => 0,
            LogicalOperator::Subquery { inner, .. } => {
                // Subqueries have materialization cost
                let inner_cost = self.estimate_logical_cost(inner);
                let inner_card = self.estimate_output_cardinality_from_logical(inner);
                // Add materialization overhead (storing results)
                inner_cost + inner_card
            }
            LogicalOperator::Bind { input, arguments, .. } => {
                let base_cost = self.estimate_logical_cost(input);
                let cardinality = self.estimate_output_cardinality_from_logical(input);
                // Add cost proportional to number of arguments and cardinality
                base_cost + (cardinality * arguments.len() as u64)
            }
            LogicalOperator::Values { values, .. } => {
                // VALUES has very low cost
                values.len() as u64
            }
            LogicalOperator::MLPredict { input, input_variables, model_name , ..} => {
                let base_cost = self.estimate_logical_cost(input);
                let cardinality = self.estimate_output_cardinality_from_logical(input);
                let ml_overhead = 100; // Cost per prediction

                match model_name {
                    ModelGetter::MLPredict(..) => {
                        // ML operations are expensive, so we add significant overhead
                        // ML prediction cost: base cost + (cardinality * input_features * ML_overhead)
                        base_cost + (cardinality * input_variables.len() as u64 * ml_overhead)
                    }
                    ModelGetter::RunMLClause(operator) => {
                        let ml_retrieval_cost = self.estimate_logical_cost(operator);
                        let ml_cardinality = self.estimate_output_cardinality_from_logical(operator);
                        base_cost + ml_retrieval_cost + (cardinality * ml_cardinality * input_variables.len() as u64 * ml_overhead)
                    }
                }
            }
        }
    }

    /// Estimates join selectivity
    fn estimate_join_selectivity(&self, left: &LogicalOperator, right: &LogicalOperator) -> f64 {
        // Extract predicates from the join patterns
        let left_predicate = self.extract_predicate_from_plan(left);
        let right_predicate = self.extract_predicate_from_plan(right);

        // Use the actual join selectivity from database stats
        match (left_predicate, right_predicate) {
            (Some(pred), _) => self.stats.get_join_selectivity(pred),
            (None, Some(pred)) => self.stats.get_join_selectivity(pred),
            (None, None) => 0.1, // Fallback to default
        }
    }

    /// Extracts the predicate ID from a logical plan if it's a scan
    fn extract_predicate_from_plan(&self, plan: &LogicalOperator) -> Option<u32> {
        match plan {
            LogicalOperator::Scan { pattern } => {
                if let Term::Constant(pred_id) = pattern.1 {
                    Some(pred_id)
                } else {
                    None
                }
            }
            LogicalOperator::Join { left, ..  } => self.extract_predicate_from_plan(left),
            LogicalOperator::Selection { predicate, .. } => self.extract_predicate_from_plan(predicate),
            LogicalOperator::Projection { predicate, .. } => self.extract_predicate_from_plan(predicate),
            LogicalOperator::Buffer {.. } => None,
            LogicalOperator::Subquery { inner, .. } => self.extract_predicate_from_plan(inner),
            LogicalOperator::Bind { input, .. } => self.extract_predicate_from_plan(input),
            LogicalOperator::Values { .. } => None,
            LogicalOperator::MLPredict { input, .. } => self.extract_predicate_from_plan(input),
        }
    }

    /// Estimates output cardinality from a logical plan
    fn estimate_output_cardinality_from_logical(&self, logical_plan: &LogicalOperator) -> u64 {
        let cost_estimator = CostEstimator::new(&self.stats);

        match logical_plan {
            LogicalOperator::Scan { pattern } => cost_estimator.estimate_cardinality(pattern),
            LogicalOperator::Selection {
                predicate,
                condition,
            } => {
                let base_card = self.estimate_output_cardinality_from_logical(predicate);
                let selectivity = cost_estimator.estimate_selectivity(condition);
                ((base_card as f64 * selectivity) as u64).max(1)
            }
            LogicalOperator::Projection { predicate, .. } => {
                self.estimate_output_cardinality_from_logical(predicate)
            }
            LogicalOperator::Join { left, right } => {
                let left_card = self.estimate_output_cardinality_from_logical(left);
                let right_card = self.estimate_output_cardinality_from_logical(right);
                let join_selectivity = self.estimate_join_selectivity(left, right);
                ((left_card.min(right_card) as f64 * join_selectivity) as u64).max(1)
            }
            LogicalOperator::Buffer { .. } => 0,
            LogicalOperator::Subquery { inner, .. } => {
                self.estimate_output_cardinality_from_logical(inner)
            }
            LogicalOperator::Bind { input, .. } => {
                self.estimate_output_cardinality_from_logical(input)
            }
            LogicalOperator::Values { values, .. } => {
                values.len() as u64
            }
            LogicalOperator::MLPredict { input, model_name, .. } => {
                let input_cardinality = self.estimate_output_cardinality_from_logical(input);
                match model_name {
                    ModelGetter::MLPredict(..) => {
                        // ML.PREDICT doesn't change cardinality, just adds a column
                        return input_cardinality;
                    }
                    ModelGetter::RunMLClause(ml_retrieval_operator) => {
                        return input_cardinality * self.estimate_output_cardinality_from_logical(ml_retrieval_operator)
                    }
                }
            }
        }
    }

    /// Updates the optimizer's statistics
    pub fn update_stats(&mut self, database: &SparqlDatabase) {
        self.stats = Arc::new(DatabaseStats::gather_stats_fast(database));
        self.memo.clear(); // Clear memo as stats have changed
    }

    /// Sets the selected variables for the query
    pub fn set_selected_variables(&mut self, variables: Vec<String>) {
        self.selected_variables = variables;
    }

    /// Gets the current statistics
    pub fn get_stats(&self) -> &DatabaseStats {
        &self.stats
    }
}

/// Resolves a URI with prefixes
/// Originally from utils.rs, but I copied it here to make life easier
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

// Helper function to convert pattern strings to TriplePattern
// Originally from utils.rs, but I copied it here to make life easier
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

#[cfg(test)]
mod tests {
    use crate::streamertail_optimizer::optimizer;

    use super::*;
    use serde::de::Expected;
    use shared::terms::Term;

    fn create_test_optimizer() -> Streamertail {
        // Create a mock database for testing
        let database = SparqlDatabase::new();
        Streamertail::new(&database)
    }

    #[test]
    fn test_count_bound_variables_all_vars() {
        let optimizer = create_test_optimizer();
        let pattern = (
            Term::Variable("s".to_string()),
            Term::Variable("p".to_string()),
            Term::Variable("o".to_string()),
        );
        assert_eq!(optimizer.count_bound_variables(&pattern), 0);
    }

    #[test]
    fn test_count_bound_variables_some_vars() {
        let optimizer = create_test_optimizer();
        let pattern = (
            Term::Constant(1),
            Term::Variable("p".to_string()),
            Term::Variable("o".to_string()),
        );
        assert_eq!(optimizer.count_bound_variables(&pattern), 1);
    }

    #[test]
    fn test_count_bound_variables_no_vars() {
        let optimizer = create_test_optimizer();
        let pattern = (Term::Constant(1), Term::Constant(2), Term::Constant(3));
        assert_eq!(optimizer.count_bound_variables(&pattern), 3);
    }

    #[test]
    fn test_physical_plan_creation_runmlclause() {
        let database = &mut SparqlDatabase::new();
        let mut optimizer = create_test_optimizer();
        let mut stats = DatabaseStats::new();

        let turtle_data = r#"
            <http://example.org#machine101> rdf:type <http://example.org#Room> .
            <http://example.org#machine101> sensor:temperature "22.5" .
            <http://example.org#machine101> sensor:humidity "45.0" .
            <http://example.org#machine101> sensor:pressure "21.0" .
            
            <http://example.org#machine102> rdf:type <http://example.org#Room> .
            <http://example.org#machine102> sensor:temperature "23.0" .
            <http://example.org#machine102> sensor:humidity "52.0" .
            <http://example.org#machine101> sensor:pressure "22.0" .

            <http://example.org#machineML> rdf:type mls:Run .
            <http://example.org#machineML> mls:hasOutput rf_temperature_predictor .
            rf_temperature_predictor rdf:type mls:Model . 

            <http://example.org#machineML2> rdf:type mls:Run .
            <http://example.org#machineML2> mls:hasOutput gb_temperature_predictor .
            gb_temperature_predictor rdf:type mls:Model .

        "#;
        database.parse_turtle(turtle_data);

        let mut prefixes = HashMap::new();
        prefixes.insert("rdf".to_string(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string());
        prefixes.insert("mls".to_string(), "http://www.w3.org/2016/10/mls/".to_string());
        prefixes.insert("sensor".to_string(), "http://sensordom.org".to_string());

        database.set_prefixes(prefixes.clone());

        // we are working with a small dataset so step = 1
        // => scale_factor = 1
        // sample_size = 15
        // scale_factor = 1
        let stats = database.get_or_build_stats();

        let inputScan1 = LogicalOperator::scan(convert_pattern_to_triple("?machine", "sensor:temperature", "?temperature", &prefixes, database));
        let inputScan2 = LogicalOperator::scan(convert_pattern_to_triple("?machine", "sensor:humidity", "?humidity", &prefixes, database));
        let inputScan3 = LogicalOperator::scan(convert_pattern_to_triple("?machine", "sensor:pressure", "?pressure", &prefixes, database));
        let mut inputlogicalop = LogicalOperator::join(inputScan1, inputScan2);
        inputlogicalop = LogicalOperator::join(inputlogicalop, inputScan3);
        let star_join_input_lo = inputlogicalop.clone();

        let mlscan1 = LogicalOperator::scan(convert_pattern_to_triple("?r", "rdf:type", "mls:Run", &prefixes, database));
        let mlscan2 = LogicalOperator::scan(convert_pattern_to_triple("?r", "mls:hasOutput", "?mlmodel", &prefixes, database));
        let mlscan3 = LogicalOperator::scan(convert_pattern_to_triple("?mlmodel", "rdf:type", "mls:Model", &prefixes, database));

        let mut mllogicalop = LogicalOperator::join(mlscan1, mlscan2);
        mllogicalop = LogicalOperator::join(mllogicalop, mlscan3);
        let star_join_ml_lo: LogicalOperator = mllogicalop.clone();
        mllogicalop = LogicalOperator::projection(mllogicalop, Vec::from(["?mlmodel".to_string()]));

        inputlogicalop = LogicalOperator::run_ml_clause_lo(inputlogicalop, mllogicalop.clone(), Vec::from(["?temperature".to_string(), "?humidity".to_string(), "?pressure".to_string()]), "?brokenPrediction".to_string());
        inputlogicalop = LogicalOperator::join(
            inputlogicalop, 
            LogicalOperator::scan(
                convert_pattern_to_triple("?x", "hvac:hasValue", "brokenPrediction", &prefixes, database)
            )
        );
        inputlogicalop = LogicalOperator::projection(inputlogicalop, Vec::from(["?machine".to_string(), "?brokenPrediction".to_string()]));

        let star_option = optimizer.is_star_query(&star_join_input_lo);
        assert!(star_option.is_some());
        let star = star_option.unwrap();
        let star_physical = optimizer.build_star_join_from_patterns(star.clone(), &star_join_input_lo);
        
        let star_ml_option = optimizer.is_star_query(&mllogicalop);
        assert!(star_ml_option.is_some());
        let star_ml_val = star_ml_option.unwrap();
        let mut star_ml = optimizer.build_star_join_from_patterns(star_ml_val, &star_join_ml_lo);
        star_ml = PhysicalOperator::projection(star_ml, Vec::from(["?mlmodel".to_string()]));

        let model_path = optimizer.discover_model_path();
        let models = ExecutionEngine::execute_with_ids(&star_ml, &mut database.clone());
        let namelist: Vec<HashMap<String, String>> = models.clone().into_iter()
                        .map(|id_result| {
                            let dict = database.dictionary.read().unwrap();
                            let result = id_result
                            .into_iter()
                            // maps the id of a model name value to the model value itself
                            .map(|(var, id)| (var, dict.decode(id).unwrap().to_string()))
                            .collect();
                            drop(dict);
                            result
                        })
                        .collect();

        let leftPhys = optimizer.choose_best_scan(
        &convert_pattern_to_triple("?x", "hvac:hasValue", "brokenPrediction", &prefixes, database)
        );
        let mut expected_physical_operator = PhysicalOperator::run_ml_clause(star_physical, models, namelist, model_path, Vec::from(["?temperature".to_string(), "?humidity".to_string(), "?pressure".to_string()]), "?brokenPrediction".to_string());
        expected_physical_operator = PhysicalOperator::optimized_hash_join(leftPhys, expected_physical_operator);
        expected_physical_operator = PhysicalOperator::projection(expected_physical_operator, Vec::from(["?machine".to_string(), "?brokenPrediction".to_string()]));

        let produced_physical_operator = optimizer.find_best_plan_recursive(&inputlogicalop, database);
        
        assert_eq!(format!("{expected_physical_operator:?}"), format!("{produced_physical_operator:?}"));

    }
}
