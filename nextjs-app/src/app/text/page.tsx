'use client';

import { useState, ChangeEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle, Loader2, Info } from 'lucide-react';

export default function TextDemo() {
  const [aslKey, setAslKey] = useState<string>('');
  const [translation, setTranslation] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleTranslate = async () => {
    if (!aslKey.trim()) {
      setError('Please enter an ASL key');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Simulate API call with a timeout
      setTimeout(() => {
        // Dummy translation based on the ASL key
        let dummyTranslation = '';
        
        switch (aslKey.toUpperCase()) {
          case 'HELLO':
            dummyTranslation = 'Hello! A friendly greeting in ASL.';
            break;
          case 'THANK YOU':
            dummyTranslation = 'Thank you! A gesture of gratitude in ASL.';
            break;
          case 'PLEASE':
            dummyTranslation = 'Please! A polite request in ASL.';
            break;
          case 'SORRY':
            dummyTranslation = 'Sorry! An apology in ASL.';
            break;
          case 'YES':
            dummyTranslation = 'Yes! An affirmative response in ASL.';
            break;
          case 'NO':
            dummyTranslation = 'No! A negative response in ASL.';
            break;
          default:
            dummyTranslation = `This is a dummy translation for the ASL key: "${aslKey}". In a real implementation, this would be translated by our AI model.`;
        }
        
        setTranslation(dummyTranslation);
        setIsLoading(false);
      }, 1500);
    } catch (error) {
      setError('An error occurred during translation');
      setIsLoading(false);
    }
  };

  const handleKeyChange = (e: ChangeEvent<HTMLInputElement>) => {
    setAslKey(e.target.value);
    setError(null);
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>ASL Text Translation</CardTitle>
            <CardDescription>
              Enter an ASL key to get a translation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid w-full items-center gap-4">
              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>How to use</AlertTitle>
                <AlertDescription>
                  Enter an ASL key (e.g., "HELLO", "THANK YOU", "PLEASE") and click "Translate" to see a dummy translation.
                </AlertDescription>
              </Alert>
              
              <div className="flex flex-col space-y-1.5">
                <Label htmlFor="aslKey">ASL Key</Label>
                <Input
                  id="aslKey"
                  placeholder="Enter ASL key (e.g., HELLO, THANK YOU)"
                  value={aslKey}
                  onChange={handleKeyChange}
                />
              </div>
              
              {translation && (
                <div className="flex flex-col space-y-1.5">
                  <Label htmlFor="translation">Translation</Label>
                  <Textarea
                    id="translation"
                    value={translation}
                    readOnly
                    className="min-h-[100px]"
                  />
                </div>
              )}
              
              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleTranslate}
              disabled={isLoading}
              className="w-full"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Translating...
                </>
              ) : (
                'Translate'
              )}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </main>
  );
} 