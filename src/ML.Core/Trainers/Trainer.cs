using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Trainers
{
    public class Trainer<T> : MvxViewModel
        where T : DataView
    {
        private Loss _loss;

        private ObservableCollection<Metric> _metrics;
        private Model<T> _model;
        private Optimizer _optimizer;
        private Dataset<T> _trainDataset;

        private TrainPlan _trainPlan;
        private Dataset<T> _valDataset;

        public Action<Trainer<T>> AfterBatchPipeline;
        public Action<Trainer<T>> AfterEpochPipeline;

        public Action<Trainer<T>> BeforeBatchPipeline;
        public Action<Trainer<T>> BeforeEpochPipeline;
        public Action<string> Print;


        public Model<T> Model
        {
            get => _model;
            set => SetProperty(ref _model, value);
        }

        public Dataset<T> TrainDataset
        {
            get => _trainDataset;
            set => SetProperty(ref _trainDataset, value);
        }

        public Dataset<T> ValDataset
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

        public async Task Fit()
        {
            TrainDataset.Should().NotBeNull("dataset should not ne null");

            Model.PipelineDataSet(TrainDataset);

            foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                await Task.Run(() =>
                {
                    BeforeEpochPipeline?.Invoke(this);

                    var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<T> data)
                    {
                        if (data.Count == 0)
                            continue;

                        BeforeBatchPipeline?.Invoke(this);
                        var batchdataSet = data.ToDatasetNDarray();

                        var predTerms = Model.CallGraph(batchdataSet.Feature);
                        var lossTerm = Loss.GetLossTerm(predTerms, batchdataSet.Label, Model.Variables);


                        var gradient = lossTerm.Differentiate(Model.Variables, Model.WeightsArray);
                        var newWeights = Optimizer.Call(Model.Weights, np.array(gradient), e);
                        Model.UpdateWeights(newWeights);

                        AfterBatchPipeline?.Invoke(this);
                    }

                    var trainMsg = new StringBuilder($"#{e + 1:D4}\t");
                    var train_loss = UpdateLossMetric(TrainDataset);
                    trainMsg.Append($"Loss:{train_loss:F4}\t");
                    foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");

                    if (ValDataset != null)
                    {
                        var val_loss = UpdateLossMetric(ValDataset);
                        trainMsg.Append($"Val_Loss:{train_loss:F4}\t");
                        foreach (var metric in Metrics) trainMsg.Append($"Val-{metric}\t");
                    }

                    Print?.Invoke(trainMsg.ToString());

                    AfterEpochPipeline?.Invoke(this);
                });

            /// early stoping
            /// Print status of each epoch
        }

        public double UpdateLossMetric(Dataset<T> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var y_pred = Model.Call(dataview.Feature);
            var y_true = dataview.Label;
            var predterms = Model.CallGraph(dataview.Feature);
            var lossTerm = Loss.GetLossTerm(predterms, dataview.Label, Model.Variables);
            var loss = lossTerm.Evaluate(Model.Variables, Model.WeightsArray);

            Metrics.ToList().ForEach(m => m.Call(y_pred, y_true));

            return loss;
        }
    }
}