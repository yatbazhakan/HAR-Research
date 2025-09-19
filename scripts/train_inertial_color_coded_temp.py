def save_individual_run_analysis(fold_results, args, config, dataset, train_metrics, val_metrics, test_metrics=None):
    """Save detailed analysis for individual fold/subject/holdout run."""
    import json
    from datetime import datetime
    from sklearn.metrics import classification_report, confusion_matrix
    
    run_name = fold_results['run_name']
    protocol = args.protocol
    
    # Create individual results directory
    results_dir = Path(f"artifacts/individual_results/{protocol}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get activity names
    activity_names = [TARGET_ACTIVITIES[dataset.idx_to_activity[i]] for i in range(len(dataset.idx_to_activity))]
    
    # Extract predictions and targets
    train_preds = fold_results['train_predictions']
    train_targets = fold_results['train_targets'] 
    val_preds = fold_results['val_predictions']
    val_targets = fold_results['val_targets']
    
    # Compute comprehensive metrics
    train_metrics_dict = train_metrics.compute()
    val_metrics_dict = val_metrics.compute()
    
    # Create detailed results dictionary
    detailed_results = {
        'run_info': {
            'run_name': run_name,
            'protocol': protocol,
            'fold_id': fold_results['fold_id'],
            'timestamp': datetime.now().isoformat(),
            'best_epoch': fold_results['best_epoch'],
            'config': config
        },
        'performance_summary': {
            'train': {
                'loss': fold_results['final_train_loss'],
                'accuracy': fold_results['final_train_acc'],
                'f1_macro': fold_results['final_train_f1'],
                'detailed_metrics': train_metrics_dict
            },
            'validation': {
                'loss': fold_results['final_val_loss'],
                'accuracy': fold_results['final_val_acc'], 
                'f1_macro': fold_results['final_val_f1'],
                'detailed_metrics': val_metrics_dict
            }
        },
        'classification_reports': {
            'train': classification_report(train_targets, train_preds, target_names=activity_names, output_dict=True),
            'validation': classification_report(val_targets, val_preds, target_names=activity_names, output_dict=True)
        },
        'confusion_matrices': {
            'train': confusion_matrix(train_targets, train_preds).tolist(),
            'validation': confusion_matrix(val_targets, val_preds).tolist()
        }
    }
    
    # Add test results if available
    if 'test_loss' in fold_results:
        test_preds = fold_results['test_predictions']
        test_targets = fold_results['test_targets']
        test_metrics_dict = test_metrics.compute() if test_metrics else {}
        
        detailed_results['performance_summary']['test'] = {
            'loss': fold_results['test_loss'],
            'accuracy': fold_results['test_acc'],
            'f1_macro': fold_results['test_f1'],
            'detailed_metrics': test_metrics_dict
        }
        detailed_results['classification_reports']['test'] = classification_report(
            test_targets, test_preds, target_names=activity_names, output_dict=True
        )
        detailed_results['confusion_matrices']['test'] = confusion_matrix(test_targets, test_preds).tolist()
    
    # Convert NumPy types for JSON serialization
    detailed_results = convert_numpy_types(detailed_results)
    
    # Save detailed results JSON
    results_file = results_dir / f"{run_name}_detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save confusion matrices as images
    save_individual_confusion_matrices(fold_results, activity_names, results_dir)
    
    # Save predictions and targets for further analysis
    save_predictions_and_targets(fold_results, results_dir)
    
    logger.info(f"Individual analysis saved for {run_name}:")
    logger.info(f"  - Results: {results_file}")
    logger.info(f"  - Plots: {results_dir}/{run_name}_*.png")

def save_individual_confusion_matrices(fold_results, activity_names, results_dir):
    """Save confusion matrices for individual run."""
    run_name = fold_results['run_name']
    
    # Train confusion matrix
    train_cm = confusion_matrix(fold_results['train_targets'], fold_results['train_predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, yticklabels=activity_names)
    plt.title(f'Training Confusion Matrix - {run_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(results_dir / f"{run_name}_train_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Validation confusion matrix
    val_cm = confusion_matrix(fold_results['val_targets'], fold_results['val_predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names)
    plt.title(f'Validation Confusion Matrix - {run_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(results_dir / f"{run_name}_val_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test confusion matrix (if available)
    if 'test_predictions' in fold_results:
        test_cm = confusion_matrix(fold_results['test_targets'], fold_results['test_predictions'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=activity_names, yticklabels=activity_names)
        plt.title(f'Test Confusion Matrix - {run_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(results_dir / f"{run_name}_test_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

def save_predictions_and_targets(fold_results, results_dir):
    """Save raw predictions and targets for further analysis."""
    run_name = fold_results['run_name']
    
    # Save as compressed numpy files
    np.savez_compressed(
        results_dir / f"{run_name}_predictions_targets.npz",
        train_predictions=fold_results['train_predictions'],
        train_targets=fold_results['train_targets'],
        val_predictions=fold_results['val_predictions'],
        val_targets=fold_results['val_targets'],
        test_predictions=fold_results.get('test_predictions', np.array([])),
        test_targets=fold_results.get('test_targets', np.array([]))
    )

