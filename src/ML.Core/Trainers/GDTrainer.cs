﻿using System.Collections.ObjectModel;
using System.Text;
using AutoDiff;
using CommunityToolkit.Mvvm.ComponentModel;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Trainers
{
    public class GDTrainer : ObservableObject
    {
        private int _batch;
        private int _currentBatchIndex;
        private int _currentEpoch;


        private Loss _loss;
        private ObservableCollection<Metric> _metrics;
        private IModelGD _modelGd;
        private Optimizer _optimizer;

        private Dataset<DataView> _trainDataset;
        private TrainPlan _trainPlan;
        private Dataset<DataView> _valDataset;


        public Action<string> Print;

        public GDTrainer()
        {
            ModelGd = new MultipleLinearRegression();
            Optimizer = new Adam();
            Loss = new MeanSquared();
            Metrics = new ObservableCollection<Metric> {new MeanSquaredError(), new RSquared()};
            TrainPlan = new TrainPlan();
        }


        public IModelGD ModelGd
        {
            get => _modelGd;
            set => SetProperty(ref _modelGd, value);
        }

        public Dataset<DataView> TrainDataset
        {
            get => _trainDataset;
            set => SetProperty(ref _trainDataset, value);
        }

        public Dataset<DataView> ValDataset
        {
            get => _valDataset;
            set => SetProperty(ref _valDataset, value);
        }

        public Optimizer Optimizer
        {
            get => _optimizer;
            set => SetProperty(ref _optimizer, value);
        }

        public Loss Loss
        {
            get => _loss;
            set => SetProperty(ref _loss, value);
        }

        public ObservableCollection<Metric> Metrics
        {
            get => _metrics;
            set => SetProperty(ref _metrics, value);
        }

        public TrainPlan TrainPlan
        {
            get => _trainPlan;
            set => SetProperty(ref _trainPlan, value);
        }

        public int CurrentEpoch
        {
            get => _currentEpoch;
            set => SetProperty(ref _currentEpoch, value);
        }

        public int CurrentBatchIndex
        {
            get => _currentBatchIndex;
            set => SetProperty(ref _currentBatchIndex, value);
        }

        public int Batch
        {
            get => _batch;
            set => SetProperty(ref _batch, value);
        }


        public async Task Fit(CancellationTokenSource cancellation = null)
        {
            Batch = TrainPlan.BatchSize == 0
                ? 1
                : TrainDataset.Count / TrainPlan.BatchSize + TrainDataset.Count % TrainPlan.BatchSize == 0
                    ? 0
                    : 1;
            CurrentEpoch = 0;

            TrainDataset.Should().NotBeNull("dataset should not ne null");

            ModelGd.PipelineDataSet(TrainDataset);

            foreach (var e in Enumerable.Range(1, TrainPlan.Epoch))
            {
                CurrentEpoch = e;
                CurrentBatchIndex = 0;
                if (cancellation?.IsCancellationRequested == true)
                    return;

                await Task.Delay(TimeSpan.FromMilliseconds(10));

                var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                while (iEnumerator.MoveNext() &&
                       iEnumerator.Current is Dataset<DataView> data)
                {
                    if (cancellation?.IsCancellationRequested == true)
                        return;

                    if (data.Count == 0)
                        continue;

                    CurrentBatchIndex++;
                    var batchdataSet = data.ToDatasetNDarray();

                    var predTerms = ModelGd.CallGraph(batchdataSet.Feature);
                    var lossTerm = Loss.GetLossTerm(predTerms, batchdataSet.Label, ModelGd.Variables);

                    var weight = ModelGd.Weights.copy();
                    var gradientArray = lossTerm.Differentiate(ModelGd.Variables, weight.GetData<double>());
                    var g = np.array(gradientArray);
                    var grad = g.reshape(weight.shape);
                    /// Todo
                    ModelGd.Weights = Optimizer.Call(weight, grad, e - 1);
                }


                var trainMsg = new StringBuilder($"#{e:D4}\t");
                var train_loss = UpdateLossMetric(TrainDataset, true);
                trainMsg.Append($"[Loss]:{train_loss:F4}\t");

                if (ValDataset == null)
                    continue;
                if (ValDataset.Value == null)
                    continue;
                if (ValDataset.Value.Length == 0)
                    continue;

                var val_loss = UpdateLossMetric(ValDataset, false);
                trainMsg.Append($"\tVal\t[Loss]:{val_loss:F4}\t");
                foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");


                Print?.Invoke(trainMsg.ToString());
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        private double UpdateLossMetric(Dataset<DataView> dataset, bool isTrain)
        {
            var dataview = dataset.ToDatasetNDarray();
            var y_pred = ModelGd.Call(dataview.Feature);
            var y_true = dataview.Label;

            var loss = Loss.GetLoss(y_pred, y_true);
            Metrics.ToList().ForEach(m => m.Call(y_true, y_pred));

            return loss;
        }
    }
}