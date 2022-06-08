using System;
using System.Windows;
using GalaSoft.MvvmLight.Ioc;

namespace ML.Guide
{
    public class ViewModelLocator
    {
        public ViewModelLocator()
        {
            /// 否则 design time 模式下，会重复注册Instance.
            SimpleIoc.Default.Reset();
        }

        public static ViewModelLocator Instance => new Lazy<ViewModelLocator>(() =>
            Application.Current.TryFindResource("Locator") as ViewModelLocator).Value;
    }
}