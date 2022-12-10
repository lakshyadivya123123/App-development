import 'package:flutter/material.dart';
class HomePage extends StatelessWidget {
    @override
  Widget build(BuildContext context) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: GridView.extent(
            primary: false,
            padding: const EdgeInsets.all(16),
            crossAxisSpacing: 10,
            mainAxisSpacing: 10,
            maxCrossAxisExtent: 200.0,
            children:<Widget> [
              Container(
                padding: const EdgeInsets.all(8),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                  Text('PACIFIC OCEAN',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 30)),
                    Icon(
                      Icons.delete,
                      size: 25.0,
                    ),
                ],
              ),
                color: Colors.yellow,
              ),
              Container(
                  padding: const EdgeInsets.all(8),
                  child: Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                   Text('INDIAN OCEAN',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 30)),
                Icon(
                  Icons.delete,
                  size: 25.0,
                ),
                ],
                  ),
              color: Colors.green,
              ),
              Container(
                padding: const EdgeInsets.all(8),
                child: Column(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                    Text('ARCTIC OCEAN',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 30)),
                Icon(
                  Icons.delete,
                  size: 25.0,
                ),
                ],
                ),
                color: Colors.green,
              ),
              Container(
                  padding: const EdgeInsets.all(8),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children:[

                    Text('ANTARTIC OCEAN',
                        textAlign: TextAlign.center,
                        style: TextStyle(fontSize: 30)),
                Icon(
                  Icons.delete,
                  size: 25.0,
                ),
                ],
                  ),
                  color: Colors.tealAccent
              ),
              Container(
                  padding: const EdgeInsets.all(8),
                  child:Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children:[
                      Text('ATLANTIC OCEAN',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 30)),
                Icon(
                  Icons.delete,
                  size: 25.0,
                ),
                ],
                  ) ,
                  color: Colors.deepOrange
              ),
            ],
          ),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: () {},
          child: Text('Click'),
        ),
        );
  }
}

class ServicesPage extends StatelessWidget {
@override
Widget build(BuildContext context) {
  return Scaffold(
    body: Center(
     child:
     Column(
       mainAxisAlignment: MainAxisAlignment.spaceEvenly,
       crossAxisAlignment: CrossAxisAlignment.center,
       children:<Widget>[
         Text(
           '...............................................................................................',
           style: TextStyle(
             color: Colors.white,
             decoration: TextDecoration.overline,
             decorationColor: Colors.black,
           ),

         ),
         
     Container(
        padding: const EdgeInsets.all(16),
        height: 300,
        width: 300,
        decoration: BoxDecoration(
          color: Colors.green,
              borderRadius: BorderRadius.circular(12)
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: <Widget>[
            Text(
              'OCEANS:',
              style: TextStyle(
                fontSize: 30,
                fontWeight: FontWeight.bold,
                color: Colors.white,),
            ),
         Text('NAME ALL THE OCEANS IN THE WORLD',
         style:TextStyle(
           fontSize: 20,
           color: Colors.black
         )

         ),
         ],
        )
      ),
  Row(
    mainAxisAlignment:  MainAxisAlignment.spaceEvenly,
    children: <Widget>[

      TextButton(
          child: Text('Show Answer'),
          onPressed:(){},
        style: TextButton.styleFrom(
          primary: Colors.white,
          backgroundColor: Colors.black,
          textStyle: TextStyle(
            fontSize: 20,
          ),
        ),
  ),
      Icon(
        Icons.navigate_next,
        size: 50,

      ),

    ],
  )

  ],)



    ),


  );
}
}




