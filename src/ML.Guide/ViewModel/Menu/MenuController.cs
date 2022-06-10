﻿using System;
using System.Threading;
using System.Windows;
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

        public LoadConfig LoadConfig { set; get; } = new();

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
            var alldataset = TextLoader.LoadDataSet(openFileDialog.FileName, datatype,
                LoadConfig.SplitChar.ToCharArray(), LoadConfig.HasHead);
            (GDTrainer.TrainDataset, GDTrainer.ValDataset) =
                LoadConfig.SplitTrainAndVal ? alldataset.Split(LoadConfig.SplitRatio) : alldataset.Split(1);
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
            GDTrainer.ValDataset = TextLoader.LoadDataSet(openFileDialog.FileName, datatype,
                LoadConfig.SplitChar.ToCharArray(), LoadConfig.HasHead);
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
                await Application.Current.Dispatcher.InvokeAsync(async () =>
                    await GDTrainer.Fit(CancellationTokenSource));
            }
            catch (Exception ex)
            {
                ;
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