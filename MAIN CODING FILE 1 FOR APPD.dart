
import 'package:flutter/material.dart';

import 'sample.dart';

void main()=> runApp(MaterialApp(
  home: BottomMenu(),
));

class BottomMenu extends StatefulWidget {
      @override
  State<StatefulWidget> createState() {
return _BottomMenuState();
  }
}
class _BottomMenuState extends State<BottomMenu> {
  var _pagesData = [ HomePage(), ServicesPage()];
  int _selectedItem = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
          title: Text('FlashCards'),
          centerTitle: true,
          backgroundColor: Colors.black
      ),
      body: Center(
        child: _pagesData[_selectedItem],
      ),
      bottomNavigationBar: BottomNavigationBar(
          items: <BottomNavigationBarItem>[
            BottomNavigationBarItem(icon: Icon(Icons.home), label: "Cards"),
            BottomNavigationBarItem(icon: Icon(Icons.home), label: "Practice")
          ],
          currentIndex: _selectedItem,
          onTap: (setValue) {
            setState(() {
              _selectedItem = setValue;
            });
          }


      ),


    );
  }
}



