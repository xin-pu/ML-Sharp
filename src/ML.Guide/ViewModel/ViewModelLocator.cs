using System;
using System.Windows;
using GalaSoft.MvvmLight.Ioc;
using ML.Core.Trainers;
using ML.Guide.ViewModel.Menu;

namespace ML.Guide.ViewModel
{
    public class ViewModelLocator
    {
        public ViewModelLocator()
        {
            /// 否则 design time 模式下，会重复注册Instance.
            SimpleIoc.Default.Reset();
            SimpleIoc.Default.Register(() => new GDTrainer());
            SimpleIoc.Default.Register(() => new MenuController());
            SimpleIoc.Default.Register(() => new MenuOption());
            SimpleIoc.Default.Register(() => new LossRecorder());
            SimpleIoc.Default.Register(() => new MetricRecorder());
        }

        public static ViewModelLocator Instance => new Lazy<ViewModelLocator>(() =>
            Application.Current.TryFindResource("Locator") as ViewModelLocator).Value;


        public GDTrainer GDTrainner => SimpleIoc.Default.GetInstance<GDTrainer>();
        public MenuOption MenuOption => SimpleIoc.Default.GetInstance<MenuOption>();
        public MenuController MenuController => SimpleIoc.Default.GetInstance<MenuController>();
        public LossRecorder LossRecorder => SimpleIoc.Default.GetInstance<LossRecorder>();
        public MetricRecorder MetricRecorder => SimpleIoc.Default.GetInstance<MetricRecorder>();
    }
}