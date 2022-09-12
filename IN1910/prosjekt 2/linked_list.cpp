#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

//Struct with an integer, node-pointer to the next node in the list,
//and nodepointer to the previous in the list.
struct Node{
  int value;
  Node* next;
  Node* previous;

  Node(int value) : value(value){

  }

};

class LinkedList{
private:
  Node* head;
  Node* tail;
  int size = 0;
public:
  LinkedList(){
    head = nullptr;
    tail = nullptr;
  }


//returnerer lengden
  int length(){
    return size;
  }

//legger til tall i listen
  void append(int value){
    size++;
    if (head == nullptr){
      head = new Node(value);
      return;
    }

    Node* current = head;
    while (current -> next != nullptr){
      current = current -> next;
      cout << "hello" << endl;
    }

    current -> next = new Node(value);

  }

  void print(){

    Node* current = head;
    cout << "[";
    while (current -> next != nullptr){
      cout << current -> value;
      cout << ", ";
      current = current-> next;
    }
    cout << current -> value << "]" << endl;
  }

  int& operator[](int x){
    if (0 <= x and x < size) {
      Node*  current = tail;
      int n = 0;
      while (n <= x);
      {
        current = current->next;
        n++;
      }
      return current->value;
      
    } else {
      throw out_of_range("range_error");
    }
  }

  void remove(int index) {
    if (0 > index or index > size) {
      throw out_of_range("range_error");
    }
    Node* current = head;
    for (int i = 0 ; i < index ; i++) {
      current =current->next;
    }
  
    Node* previous = current->previous;
    Node* next = current->next;
    delete[] current;
    previous->next = next;
    next->previous = previous;
    size--;
  }

};

int main(){

  LinkedList example;
  example.append(2);
  example.append(3);
  example.print();

  return 0;
}
