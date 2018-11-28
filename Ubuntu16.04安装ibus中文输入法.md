ibus输入法的安装  
1.安装语言包  
System Settings–>LanguageSupport–>Install/RemoveLanguages选中chinese  
点击Apply应用即可，等待下载安装完成。  
2.安装ibus框架  
sudo apt-get install ibus ibus-clutter  
sudo apt-get install ibus-gtkibus-gtk3 （这个安装不安装无所谓）  
sudo apt-get install ibus-qt4  
（启动ibus框架：im-config-s ibus或者im-config。）  

3.安装拼音引擎
sudo apt-get install  ibus-libpinyin  

这一步做完，要重启系统！！！  
4.设置ibus框架
sudo ibus-setup  

5.选中inputmethod，Add刚才安装的中文拼音。  

6.添加输入法  
在System Settings-->TextEntry,点击左下角的+号，选择安装的输入法即可完成安装。