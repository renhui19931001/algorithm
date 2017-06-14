#include <iostream>
#include <vector>
using namespace std;
void shell_sort(int*,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	shell_sort(a,length);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void shell_sort(int a[],int n){
	for(int gas = 3;gas>=1;gas--){
		for(int i = 0;i <gas;i++){
			for(int j = i+gas;j<n;j++){
				int k = j-gas;
				int zhi = a[j];
				while(k>= 0 && a[k] > zhi){
					a[k+gas] = a[k];
					k = k - gas;
				}
				a[k+gas] = zhi;
			}
		}
	}
}