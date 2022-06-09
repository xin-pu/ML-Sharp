using System;
using System.Threading;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using Microsoft.Win32;
using ML.Core.Data.Loader;
using ML.Core.Trainers;

namespace ML.Guide.ViewModel.Menu
{
    public class MenuController : ViewModelBase
    {
        public GDTrainer GDTrainer => ViewModelLocator.Instance.GDTrainner;

        #region DataSet Command

        public RelayCommand<Type> LoadTrainDatasetCommand => new(datatype => LoadTrainDatasetCommand_Execute(datatype));

        public RelayCommand<Type> LoadValDatasetCommand => new(datatype => LoadValDatasetCommand_Execute(datatype));


        private void LoadTrainDatasetCommand_Execute(Type datatype)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = @"txt(*.txt)|*.txt"
            };
            var res = openFileDialog.ShowDialog();
            if (res != true || openFileDialog.FileName == "")
                return;
            var alldataset = TextLoader.LoadDataSet(openFileDialog.FileName, datatype, false);
            (GDTrainer.TrainDataset, GDTrainer.ValDataset) =
                SplitTrainAndVal ? alldataset.Split(SplitRatio) : alldataset.Split(0);
        }

        private void LoadValDatasetCommand_Execute(Type datatype)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = @"txt(*.txt)|*.txt"
            };
            var res = openFileDialog.ShowDialog();
            if (res != true || openFileDialog.FileName == "")
                return;
            GDTrainer.ValDataset = TextLoader.LoadDataSet(openFileDialog.FileName, datatype, false);
        }

        private bool _splitTrainAndVal = true;
        private double _splitRatio = 0.8;

        public bool SplitTrainAndVal
        {
            get => _splitTrainAndVal;
            set => Set(ref _splitTrainAndVal, value);
        }


        public double SplitRatio
        {
            get => _splitRatio;
            set => Set(ref _splitRatio, value);
        }

        #endregion


        #region Control Command

        public RelayCommand TrainCommand => new(() => TrainCommand_Execute());
        public RelayCommand CancelCommand => new(() => CancelCommand_Execute());
        public CancellationTokenSource CancellationTokenSource { internal set; get; } = new();

        private async void TrainCommand_Execute()
        {
            CancellationTokenSource = new CancellationTokenSource();

            try
            {
                PreCheck();
                await GDTrainer.Fit(CancellationTokenSource);
            }
            catch (Exception)
            {
                // ignored
            }
        }


        public void PreCheck()
        {
        }


        private void CancelCommand_Execute()
        {
            CancellationTokenSource.Cancel();
        }

        #endregion
    }
}