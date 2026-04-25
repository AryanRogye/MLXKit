//
//  FileDownloader.swift
//  ModelChat
//
//  Created by Aryan Rogye on 4/5/26.
//

import Foundation

final class FileDownloader: NSObject, URLSessionDownloadDelegate {
    
    var onProgress: ((Double) -> Void)?
    
    private var continuations: [URLSessionTask: CheckedContinuation<Void, Error>] = [:]
    private var destinations: [URLSessionTask: URL] = [:]
    private let lock = NSLock()
    
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 300
        config.waitsForConnectivity = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()
    
    func download(from remoteURL: URL, to localURL: URL, retries: Int = 3) async throws {
        var lastError: Error?
        for attempt in 0..<retries {
            do {
                if attempt > 0 {
                    try await Task.sleep(nanoseconds: UInt64(attempt) * 1_000_000_000)
                }
                try await _download(from: remoteURL, to: localURL)
                return
            } catch {
                lastError = error
            }
        }
        throw lastError!
    }
    
    private func _download(from remoteURL: URL, to localURL: URL) async throws {
        // Create intermediate directories if needed (e.g. for sharded files in subfolders)
        let dir = localURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let task = session.downloadTask(with: remoteURL)
            lock.lock()
            continuations[task] = cont
            destinations[task] = localURL
            lock.unlock()
            task.resume()
        }
    }
    
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress?(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
    
    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        lock.lock()
        let cont = continuations.removeValue(forKey: downloadTask)
        let dest = destinations.removeValue(forKey: downloadTask)
        lock.unlock()
        
        guard let dest else {
            cont?.resume(throwing: URLError(.badServerResponse))
            return
        }
        
        do {
            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.moveItem(at: location, to: dest)
            cont?.resume()
        } catch {
            cont?.resume(throwing: error)
        }
    }
    
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }
        lock.lock()
        let cont = continuations.removeValue(forKey: task)
        destinations.removeValue(forKey: task)
        lock.unlock()
        cont?.resume(throwing: error)
    }
}
