import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";

export default function Home() {
  return (
    <div className="container mx-auto p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">Welcome to ASL Translation Platform</h1>
        <p className="text-xl text-gray-600 mb-8">
          Transform American Sign Language into text in real-time using our advanced AI technology.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Video Translation</CardTitle>
              <CardDescription>
                Use your camera to translate ASL signs in real-time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-500">
                Perfect for live conversations and learning ASL. Our AI model can detect and translate hand gestures with high accuracy.
              </p>
            </CardContent>
            <CardFooter>
              <Link href="/video" className="w-full">
                <Button className="w-full">Try Video Demo</Button>
              </Link>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Text Translation</CardTitle>
              <CardDescription>
                Convert text to ASL descriptions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-500">
                Get detailed descriptions of how to perform ASL signs for any text input. Great for learning and practice.
              </p>
            </CardContent>
            <CardFooter>
              <Link href="/text" className="w-full">
                <Button className="w-full">Try Text Demo</Button>
              </Link>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
} 