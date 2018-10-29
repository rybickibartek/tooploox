#include <stdio.h>

size_t my_123_w_strlen(const char *s)
{
   char *p = s;

   while (*p)
      ++p;

   return (p - s);
}

wchar_t *my_wstrchr(wchar_t *ws, wchar_t wc)
{
   while (*ws) 
   {
      if (*ws == wc)
      return ws;
      ++ws;
   }
   return NULL;
   
}

char *my_strcpy(char *t, char *s)
{
   char *p = t;   
 
   while (*t++ = *s++);

   return p;
}

int main(void)
{
   int i;
   char *s[] = 
   {
      "Git tutorials",
      "Tutorials Point"
   };

   for (i = 0; i < 2; ++i)
      
   printf("string lenght of %s = %d\n", s[i], my_strlen(s[i]));

   printf("%s\n", my_strcpy(p1, "Hello, World !!!"));

   return 0;
}
