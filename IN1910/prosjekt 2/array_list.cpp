#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

//Lager en non-fixed array klasse
class ArrayList{

//Konstruktør
private:
  int *m_data;
  int m_capacity = 1;
  int m_size = 0;
  int m_growth = 2;

//Lager en resize klasse. Denne funksjonen alokerer en ny array, og dobler kapasiteten med to.
//Så kopierer den all data fra den tidligere arrayet til den nye
  void resize(){
    int capacity = m_growth * m_capacity;
    int *data = new int[capacity];
    for (int i = 0; i < m_capacity; i++){
      data[i] = m_data[i];
    }
    delete[] m_data;
    m_data = data;
    m_capacity = capacity;
  }

  //
  void shrink_to_fit(){

    int n;
    for (n = 1 ; n < m_size  ; n++){
      n = n * 2;
    }
    int closest = n;

    for (int i = 0; closest < m_capacity ; i++){
      m_capacity = m_capacity / (2*i);
    }
  }

public:

  ArrayList(vector<int> vec){
    m_data = new int[m_capacity];
    for (int i: vec){
      append(i);
    }
  }

  ArrayList(){
    m_data = new int[m_capacity];

  }

  ~ArrayList(){
    delete[] m_data;
  }

//returnerer lengden
  int length(){
    return m_size;
  }

//legger til verdien i listen
  void append(int x){
    if (m_size >= m_capacity) {
      resize();
    }
    m_data[m_size] = x;
    m_size++;
  }

//printer listen
  void print(){
    cout << "[";
    for (int i = 0; i < m_size - 1; i++){
      cout << m_data[i] << ", ";
    }
    cout << m_data[m_size - 1] << "]\n";
  }

  int& operator[](int x){
    if (0 <= x and x < m_size) {
      return m_data[x];
    } else {
      throw out_of_range("range_error");
    }
  }

// legges til et tall i lista, og resten av tallene skal flyttes et plass videre
  void insert(int val, int index){
    m_size ++;

    for (int n = 1; n < (m_size - index); n++){
      m_data[m_size-n] = m_data[m_size-n-1];
    }

    m_data[index] = val;
  }

  //fjerner et element
  void remove(int index){
    for (int n = index; n < (m_size);n++){
      m_data[n] = m_data[n+1];
    }
    m_size --;

    if (m_size < m_capacity/4){
      shrink_to_fit();
    }

  }

  void remove(){
    m_size --;
    if (m_size < m_capacity/4){
      shrink_to_fit();
    }
  }

  int pop(int index){
    int d = m_data[index];
    for (int n = index; n < (m_size);n++){
      m_data[n] = m_data[n+1];
    }
    m_size --;

    if (m_size < m_capacity/4){
      shrink_to_fit();
    }

    return d;
  }

  int pop(){
    int d = m_data[m_size-1];

    m_size --;

    if (m_size < m_capacity/4){
      shrink_to_fit();
    }

    return d;
  }


};

bool is_prime(int n){
  //sjekker om tallet er et primtall
  if (n == 1) {
    return false;
  }

  for (int i=2; i<n; i++){
    if(n % i == 0) {
      return false;
    }
  }
  return true;
}


int main(){
  ArrayList arr{};
  int n = 1;
  while (arr.length() < 10){
    if (is_prime(n)){
      arr.append(n);
    }
    n++;
  }
  arr.print();

  ArrayList array_vec({0,2,0,4,0,6});
  array_vec[0] = 1;
  array_vec[2] = 2;
  array_vec[4] = 5;
  array_vec.print();

  ArrayList example({8,7,6,5,4,3});
  example.insert(0,2);
  example.print();

  ArrayList example1({0,2,4,6,8,10});
  example1.pop(0);
  example1.print();

  return 0;
}
